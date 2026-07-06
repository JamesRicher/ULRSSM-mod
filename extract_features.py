import argparse
import glob
import os
import numpy as np
import torch

from networks.diffusion_network import DiffusionNet
from utils.shape_util import read_shape


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract per-vertex DiffusionNet features for a dataset under ../data.'
    )
    parser.add_argument('--dataset', default='FAUST_r',
                        help='Dataset name (folder under --data_root). May include a subset, '
                             'e.g. "SHREC16/cuts".')
    parser.add_argument('--data_root',
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                        help='Root directory containing the dataset folders '
                             '(default: ../data relative to this script).')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint. If omitted, inferred from --dataset.')
    parser.add_argument('--mesh_subdir', default='off',
                        help='Subdirectory containing mesh files (default: off).')
    parser.add_argument('--mesh_ext', default='off',
                        help='Mesh file extension to load (off or obj). Default: off.')
    parser.add_argument('--output_subdir', default='feats',
                        help='Subdirectory (under the dataset folder) to write features into.')
    parser.add_argument('--input_type', default='wks', choices=['wks', 'xyz'],
                        help='DiffusionNet input feature type.')
    parser.add_argument('--out_channels', type=int, default=256,
                        help='Number of output feature channels.')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip L2-normalization of per-vertex features.')
    return parser.parse_args()


def infer_checkpoint(dataset):
    """Best-effort mapping from dataset folder to default checkpoint."""
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


def main():
    args = parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    mesh_dir = os.path.join(dataset_dir, args.mesh_subdir)
    output_dir = os.path.join(dataset_dir, args.output_subdir)
    checkpoint = args.checkpoint or infer_checkpoint(args.dataset)

    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f'Mesh directory not found: {mesh_dir}')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    in_channels = 128 if args.input_type == 'wks' else 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    print(f'Dataset:    {args.dataset}')
    print(f'Mesh dir:   {mesh_dir}')
    print(f'Output dir: {output_dir}')
    print(f'Checkpoint: {checkpoint}')

    feature_extractor = DiffusionNet(
        in_channels=in_channels,
        out_channels=args.out_channels,
        input_type=args.input_type,
    ).to(device)
    feature_extractor.load_state_dict(
        torch.load(checkpoint, map_location=device)['networks']['feature_extractor'],
        strict=True,
    )
    feature_extractor.eval()

    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, f'*.{args.mesh_ext}')))
    print(f'Found {len(mesh_files)} *.{args.mesh_ext} meshes in {mesh_dir}')

    for i, path in enumerate(mesh_files):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f'{name}.npy')

        vert_np, face_np = read_shape(path)
        vert = torch.from_numpy(vert_np).to(device=device, dtype=torch.float32)
        face = torch.from_numpy(face_np).to(device=device, dtype=torch.long)

        with torch.no_grad():
            feat = feature_extractor(vert.unsqueeze(0), face.unsqueeze(0))
        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)
        if not args.no_normalize:
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12)

        np.save(out_path, feat)
        print(f'[{i + 1}/{len(mesh_files)}] {name}: features {feat.shape} -> {out_path}')

    print(f'Done. Features saved to {output_dir}/')


if __name__ == '__main__':
    main()
