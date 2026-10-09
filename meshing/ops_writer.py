"""
Streaming OPS/OpenSBLI grid writer for extruded multiblock grids.

Each block is a 2D (x, y) plane extruded in z, with z allowed to vary with x
(periodic-uniform or sidewall-stretched spans). The 3D coordinate arrays are never
built in memory: halos are extrapolated in 2D and the file is written one k-slab at a time.

File layout matches the legacy Py_to_HDF5.arrays_to_HDF5 writer:
    /opensbliblock{NN}/x0_B{N}, x1_B{N}, x2_B{N}[, h0_B{N}]
    dataset shape (nz + 2h, ny + 2h, nx + 2h), float64, linear-extrapolated halos
    OPS attributes: d_p, d_m, dim, ops_type, block_index, base, type, block, size
Differences from the legacy writer (extend_halo_3D):
    - y and z halos are genuinely linear (legacy set all h layers to the same value,
      i.e. zero spacing between halo layers; only its x halos were correct)
    - halo edges/corners are filled (legacy left them as 0.0)
    - root attributes carry provenance (mesh config, geometry hash)
"""
import json
import time

import h5py
import numpy as np


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extrap_1d_axis(a, h, axis):
    """Linear extrapolation of h points on both ends of `axis` (slope from the two end points)."""
    a = np.moveaxis(np.asarray(a, float), axis, 0)
    n = a.shape[0]
    out = np.empty((n + 2 * h,) + a.shape[1:])
    out[h:h + n] = a
    dl, dr = a[1] - a[0], a[-1] - a[-2]
    for m in range(1, h + 1):
        out[h - m] = a[0] - m * dl
        out[h + n - 1 + m] = a[-1] + m * dr
    return np.moveaxis(out, 0, axis)


def pad_2d(a, h):
    """Pad a (ny, nx) plane in x then y -> (ny + 2h, nx + 2h), corners filled consistently."""
    return extrap_1d_axis(extrap_1d_axis(a, h, axis=1), h, axis=0)


def _block_attrs(group, ib, ndim=3):
    group.attrs.create("dims", [ndim], dtype="int32")
    group.attrs.create("ops_type", "ops_block", dtype="S8")      # legacy truncation kept on purpose
    group.attrs.create("index", [ib], dtype="int32")


def _dset_attrs(dset, ib, size, h, ndim=3):
    dset.attrs.create("d_p", [h] * ndim, dtype="int32")
    dset.attrs.create("d_m", [-h] * ndim, dtype="int32")
    dset.attrs.create("dim", [1], dtype="int32")
    dset.attrs.create("ops_type", "ops_dat", dtype="S10")
    dset.attrs.create("block_index", [ib], dtype="int32")
    dset.attrs.create("base", [0] * ndim, dtype="int32")
    dset.attrs.create("type", "double", dtype="S15")
    dset.attrs.create("block", "opensbliblock%02d" % ib, dtype="S25")
    dset.attrs.create("size", size, dtype="int32")


def write_extruded_blocks(path, blocks, nhalo=5, scale=1.0, h0=True, provenance=None):
    """
    blocks : {ib: (X, Y, Z)} with X, Y of shape (ny, nx) and Z of shape (nz, nx)
             (z may vary with x; it is constant along y). Physical units, divided by `scale`.
    """
    h = nhalo
    t0 = time.time()
    with h5py.File(path, "w") as f:
        if provenance:
            f.attrs["provenance_json"] = json.dumps(provenance)
        for ib, (X, Y, Z) in sorted(blocks.items()):
            ny, nx = X.shape
            nz = Z.shape[0]
            assert Y.shape == (ny, nx) and Z.shape[1] == nx, f"B{ib}: shape mismatch"
            size = [nx, ny, nz]                                  # OPS order
            shape = (nz + 2 * h, ny + 2 * h, nx + 2 * h)
            _log(f"B{ib}: (nz, ny, nx) = ({nz}, {ny}, {nx})  -> {np.prod(shape) / 1e6:.1f}M per variable")

            g = f.create_group("opensbliblock%02d" % ib)
            _block_attrs(g, ib)

            Xp = pad_2d(X / scale, h)                            # (ny+2h, nx+2h)
            Yp = pad_2d(Y / scale, h)
            Zp = extrap_1d_axis(extrap_1d_axis(Z / scale, h, axis=1), h, axis=0)   # (nz+2h, nx+2h)
            H0p = pad_2d(np.broadcast_to(Y.min(axis=0) / scale, (ny, nx)), h) if h0 else None

            names = ["x0", "x1", "x2"] + (["h0"] if h0 else [])
            dsets = {}
            for nm in names:
                d = g.create_dataset(f"{nm}_B{ib}", shape=shape, dtype="f8")
                _dset_attrs(d, ib, size, h)
                dsets[nm] = d

            slab = np.empty(shape[1:])
            for k in range(shape[0]):                            # stream k-slabs
                dsets["x0"][k] = Xp
                dsets["x1"][k] = Yp
                slab[:] = Zp[k][None, :]
                dsets["x2"][k] = slab
                if h0:
                    dsets["h0"][k] = H0p
            f.flush()
    _log(f"wrote {path}  ({time.time() - t0:.1f} s)")


def read_plane(path, ib, k=0, nhalo=5):
    """Interior (x, y) of one k-plane - cheap readback for checking."""
    h = nhalo
    with h5py.File(path, "r") as f:
        g = f["opensbliblock%02d" % ib]
        x = g[f"x0_B{ib}"][h + k, h:-h, h:-h]
        y = g[f"x1_B{ib}"][h + k, h:-h, h:-h]
    return x, y