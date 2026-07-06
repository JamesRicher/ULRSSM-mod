# Code Citations

## License: MIT
https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching/blob/3d47af4982f3e26c7d14861f04904d0123004bda/utils/texture_util.py

```
The texture warping is likely due to how UV coordinates are being generated in `generate_tex_coords()`.

Looking at [utils/texture_util.py](utils/texture_util.py#L18-L27):

```python
def generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    ind = np.argsort(np.std(verts, axis=0))[::-1]
    verts = verts[:, ind]
    vt = np.stack([verts[:, col1], verts[:, col2]], axis=-1)
    vt -= np.min(vt, axis=0, keepdims=True)
    vt = mult_const * vt / np.max(vt)
    ...
```

This function creates UVs by **orthographic projection** of the mesh onto the plane defined by its two highest-variance axes. This is extremely crude because:

1. **Multiple 3D points map to same UV** — two distant parts of a curved shape can project to the same (u,v) coordinate, causing texture overlap
2. **No unwrapping** — complex 3D shapes (humans, animals) can't be flattened to 2D without significant distortion
3. **Amplifies across correspondence** — when the warped UV is transferred to the target mesh, the distortion may propagate

---

## Why
```

