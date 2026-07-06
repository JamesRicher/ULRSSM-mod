import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from networks.diffusion_network import DiffusionNet
from utils.shape_util import read_shape


def infer_checkpoint(dataset):
    """Best-effort mapping from dataset folder name to default checkpoint."""
    key = dataset.lower().rstrip('/').replace('\\', '/')
    mapping = {
        'faust_r': 'faust.pth',
        'faust_a': 'faust.pth',
        'scape_r': 'scape.pth',
        'scape_a': 'scape.pth',
        'shrec19_r': 'faust_scape.pth',
        'shrec16/cuts': 'shrec16_cuts.pth',
        'shrec16/holes': 'shrec16_holes.pth',
        'shrec16/null': 'shrec16_cuts.pth',
        'shrec16_test/cuts': 'shrec16_cuts.pth',
        'shrec16_test/holes': 'shrec16_holes.pth',
        'shrec20': 'shrec20.pth',
        'smal_r': 'smal.pth',
        'dt4d_r': 'dt4d.pth',
        'topkids': 'topkids.pth',
    }
    if key in mapping:
        return os.path.join('checkpoints', mapping[key])
    # Fallback: checkpoints/<basename>.pth
    return os.path.join('checkpoints', f'{os.path.basename(key)}.pth')


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Extract per-vertex DiffusionNet features from .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--checkpoint', default=None,
                        help='network checkpoint. If omitted, inferred from the data_root folder name.')
    parser.add_argument('--input_type', default='wks', choices=['wks', 'xyz'],
                        help='DiffusionNet input feature type.')
    parser.add_argument('--out_channels', type=int, default=256,
                        help='number of output feature channels.')
    parser.add_argument('--no_normalize', action='store_true',
                        help='no L2-normalization of per-vertex features.')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    out_channels = args.out_channels
    no_normalize = args.no_normalize
    assert out_channels > 0, f'Invalid out_channels: {out_channels}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    checkpoint = args.checkpoint or infer_checkpoint(os.path.basename(os.path.normpath(data_root)))
    assert os.path.isfile(checkpoint), f'Checkpoint not found: {checkpoint}'

    feats_dir = os.path.join(data_root, 'feats')
    os.makedirs(feats_dir, exist_ok=True)

    in_channels = 128 if args.input_type == 'wks' else 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load feature extractor
    feature_extractor = DiffusionNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_type=args.input_type,
        cache_dir=os.path.join(data_root, 'diffusion'),
    ).to(device)
    feature_extractor.load_state_dict(
        torch.load(checkpoint, map_location=device)['networks']['feature_extractor'],
        strict=True,
    )
    feature_extractor.eval()

    # read .off files
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    assert len(off_files) != 0

    for off_file in tqdm(off_files):
        verts, faces = read_shape(off_file)
        filename = os.path.basename(off_file)

        vert = torch.from_numpy(verts).to(device=device, dtype=torch.float32)
        face = torch.from_numpy(faces).to(device=device, dtype=torch.long)

        with torch.no_grad():
            feat = feature_extractor(vert.unsqueeze(0), face.unsqueeze(0))
        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)

        if not no_normalize:
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12)

        # save results
        np.save(os.path.join(feats_dir, filename.replace('.off', '.npy')), feat)
