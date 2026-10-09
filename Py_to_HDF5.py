"""
Converts DAT grid coords to HDF5 format. For reading into OpenSBLI
need to check if it reads in z coords from the dat unlike the old version

    lets modify this so it is a callable funtion that reads in the arrys straight away and not dat files
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import gc
import os
from types import SimpleNamespace
import time

class SimulationBlock:
    def __init__(self, ndim, block_number=0):
        self.ndim = ndim
        self.blocknumber = block_number
        self.blockname = "opensbliblock%02d" % block_number
    def location_dataset(self, name):
        return SimpleNamespace(base="%s_B%d" % (name, self.blocknumber))


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class _Timer:
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t0 = time.time()
        _log(f"{self.label} ...")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        _log(f"{self.label} done in {dt:.2f}s")


# ---------- paths ----------
HERE = os.path.dirname(__file__)                       # directory of this script
GRID_DIR = os.path.abspath(os.path.join(HERE, "grids"))  # ./grids next to script




# Inputs
# ---------------------------------------------------------------------------------------------------------------------

h0_switch = 1


def apply_group_attributes(group, block):
    group.attrs.create("dims", [block.ndim], dtype="int32")
    group.attrs.create("ops_type", u"ops_block", dtype="S8")
    group.attrs.create("index", [block.blocknumber], dtype="int32")
    return



def set_hdf5_metadata(dset, halos, npoints, block):
    """ Function to set hdf5 metadata required by OPS to a dataset. """
    d_m = halos[0]
    d_p = halos[1]

    dset.attrs.create("d_p", d_p, dtype="int32")
    dset.attrs.create("d_m", d_m, dtype="int32")
    dset.attrs.create("dim", [1], dtype="int32")
    dset.attrs.create("ops_type", u"ops_dat", dtype="S10")
    dset.attrs.create("block_index", [block.blocknumber], dtype="int32")
    dset.attrs.create("base", [0 for i in range(block.ndim)], dtype="int32")
    dset.attrs.create("type", u"double", dtype="S15")
    dset.attrs.create("block", u"%s" % block.blockname, dtype="S25")
    dset.attrs.create("size", npoints, dtype="int32")
    return



def extend_halo_3D(arr, nhalo):
    """
    Extend a 3D array with halo regions using linear extrapolation.
    arr must be indexed [z, y, x].
    """
    nz, ny, nx = arr.shape
    padded = np.zeros((nz + 2 * nhalo, ny + 2 * nhalo, nx + 2 * nhalo))
    padded[nhalo:nhalo + nz, nhalo:nhalo + ny, nhalo:nhalo + nx] = arr
    
    

    # Z direction (axis 0)
    for k in range(nz + 2 * nhalo):

        if k < nhalo:
            padded[k, nhalo:nhalo + ny, nhalo:nhalo + nx] = (
                padded[nhalo, nhalo:nhalo+ny, nhalo:nhalo+nx] - 
                (padded[nhalo + 1, nhalo:nhalo+ny, nhalo:nhalo+nx] - padded[nhalo, nhalo:nhalo+ny, nhalo:nhalo+nx])
            )

        elif  k >= nz + nhalo:
            padded[k, nhalo:nhalo+ny, nhalo:nhalo+nx] = (
                padded[nz + nhalo - 1, nhalo:nhalo+ny, nhalo:nhalo+nx] 
                + (padded[nz + nhalo - 1, nhalo:nhalo+ny, nhalo:nhalo+nx] - padded[nz + nhalo - 2, nhalo:nhalo+ny, nhalo:nhalo+nx])
            )
    
    # Y direction (axis 1)
    for j in range(ny + 2 * nhalo):

        if j < nhalo:
            padded[nhalo:nhalo+nz, j, nhalo:nhalo+nx] = (
                padded[nhalo:nhalo+nz, nhalo, nhalo:nhalo+nx] 
                - (padded[nhalo:nhalo+nz, nhalo + 1, nhalo:nhalo+nx] - padded[nhalo:nhalo+nz, nhalo, nhalo:nhalo+nx])
            )

        elif j >= ny + nhalo:
            padded[nhalo:nhalo+nz, j, nhalo:nhalo+nx] = (
                padded[nhalo:nhalo+nz, ny + nhalo - 1, nhalo:nhalo+nx] 
                + (padded[nhalo:nhalo+nz, ny + nhalo - 1, nhalo:nhalo+nx] - padded[nhalo:nhalo+nz, ny + nhalo - 2, nhalo:nhalo+nx])
            )

    # X-direction (axis 2) — requires precomputed diffs
    left_diff = padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nhalo + 1] - padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nhalo]
    right_diff = padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nx + nhalo - 1] - padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nx + nhalo - 2]

    for i in range(nx + 2 * nhalo):

        if i < nhalo:
            padded[nhalo:nhalo+nz, nhalo:nhalo+ny, i] = (
                padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nhalo] - left_diff * (nhalo - i)
            ) 

        if i >= nx + nhalo:
            padded[nhalo:nhalo+nz, nhalo:nhalo+ny, i] = (
                padded[nhalo:nhalo+nz, nhalo:nhalo+ny, nx + nhalo - 1] 
                + right_diff * (i - (nx + nhalo - 1))
            )


    return padded

def extend_halo_2D(arr, nhalo):
    """
    Extend a 2D array with halo regions using linear extrapolation.
    arr must be indexed [y, x].
    """
    ny, nx = arr.shape
    padded = np.zeros((ny + 2 * nhalo, nx + 2 * nhalo))
    padded[nhalo:nhalo + ny, nhalo:nhalo + nx] = arr
    
    

    # Y direction (axis 0)
    for j in range(ny + 2 * nhalo):

        if j < nhalo:
            padded[j, nhalo:nhalo+nx] = (
                padded[nhalo, nhalo:nhalo+nx] 
                - (padded[nhalo + 1, nhalo:nhalo+nx] - padded[nhalo, nhalo:nhalo+nx])
            )

        elif j >= ny + nhalo:
            padded[j, nhalo:nhalo+nx] = (
                padded[ny + nhalo - 1, nhalo:nhalo+nx] 
                + (padded[ny + nhalo - 1, nhalo:nhalo+nx] - padded[ny + nhalo - 2, nhalo:nhalo+nx])
            )

    # X-direction (axis 1) — requires precomputed diffs
    left_diff = padded[nhalo:nhalo+ny, nhalo + 1] - padded[nhalo:nhalo+ny, nhalo]
    right_diff = padded[nhalo:nhalo+ny, nx + nhalo - 1] - padded[nhalo:nhalo+ny, nx + nhalo - 2]

    for i in range(nx + 2 * nhalo):

        if i < nhalo:
            padded[nhalo:nhalo+ny, i] = (
                padded[nhalo:nhalo+ny, nhalo] - left_diff * (nhalo - i)
            ) 

        if i >= nx + nhalo:
            padded[nhalo:nhalo+ny, i] = (
                padded[nhalo:nhalo+ny, nx + nhalo - 1] 
                + right_diff * (i - (nx + nhalo - 1))
            )
    return padded

def write_grid_variable_from_vectors(group, block, name, xv, yv, zv,
                                     nhalo, halos, size,
                                     k_chunk=16, dtype=np.float64,
                                     progress_every=10):
    loc = block.location_dataset(name)
    dset_name = str(getattr(loc, "base", loc))
    h = nhalo

    xv_h = extend_1d_linear(xv, h).astype(dtype, copy=False)
    yv_h = extend_1d_linear(yv, h).astype(dtype, copy=False)
    zv_h = extend_1d_linear(zv, h).astype(dtype, copy=False)

    NxF, NyF, NzF = xv_h.size, yv_h.size, zv_h.size
    shape_full = (NxF, NyF, NzF)

    # Chunking tuned for k-slab writes
    chunks = (min(128, NxF), min(128, NyF), 1)

    dset = group.create_dataset(dset_name, shape=shape_full, dtype=dtype,
                                chunks=chunks, compression=None)
    set_hdf5_metadata(dset, halos=halos, npoints=size, block=block)

    # reusable slab buffer
    slab = np.empty((NxF, NyF, min(k_chunk, NzF)), dtype=dtype)

    for it, k0 in enumerate(range(0, NzF, k_chunk)):
        k1 = min(k0 + k_chunk, NzF)
        klen = k1 - k0
        kk = slice(k0, k1)

        if name == "x0":
            slab[:, :, :klen] = xv_h[:, None, None]          # broadcasts
        elif name == "x1":
            slab[:, :, :klen] = yv_h[None, :, None]
        elif name == "x2":
            slab[:, :, :klen] = zv_h[k0:k1][None, None, :]
        elif name == "h0":
            slab[:, :, :klen] = float(np.min(yv))
        else:
            raise ValueError(f"Unknown grid variable name: {name}")

        dset[:, :, kk] = slab[:, :, :klen]

        if (it % progress_every) == 0:
            _log(f"{block.blockname} {dset_name}: k {k0}/{NzF}")

    return dset

def extend_1d_linear(v, nhalo):
    """
    Linear extrapolate a 1D vector with nhalo points on each side.
    Matches your halo logic (constant slope from first two / last two points).
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    out = np.empty(n + 2*nhalo, dtype=v.dtype)
    out[nhalo:nhalo+n] = v

    # left halos
    dl = v[1] - v[0]
    for i in range(nhalo):
        out[nhalo-1-i] = v[0] - dl*(i+1)

    # right halos
    dr = v[-1] - v[-2]
    for i in range(nhalo):
        out[nhalo+n+i] = v[-1] + dr*(i+1)

    return out



def write_blocks_to_hdf5_from_vectors(x_vec, y_vec, z_vec, nblocks, h5f, nhalo=5, h0_switch=True, k_chunk=8):
    _log(f"Writing VECTOR data to hdf5 for {nblocks} blocks")

    for block_number in range(nblocks):
        xv = x_vec[block_number]
        yv = y_vec[block_number]
        zv = z_vec[block_number]

        _log(f"--- Block {block_number} (vectors) ---")

        with _Timer("Create group + attributes"):
            ndim = 3
            b = SimulationBlock(ndim, block_number=block_number)
            g1 = h5f.create_group(b.blockname)
            apply_group_attributes(g1, b)

        halos = [[-nhalo]*3, [nhalo]*3]
        OPS_shape = [len(zv), len(yv), len(xv)]  # same convention you used before

        with _Timer("Write x0"):
            write_grid_variable_from_vectors(g1, b, "x0", xv, yv, zv, nhalo, halos, OPS_shape, k_chunk=k_chunk, dtype=np.float32)
        with _Timer("Write x1"):
            write_grid_variable_from_vectors(g1, b, "x1", xv, yv, zv, nhalo, halos, OPS_shape, k_chunk=k_chunk, dtype=np.float32)
        with _Timer("Write x2"):
            write_grid_variable_from_vectors(g1, b, "x2", xv, yv, zv, nhalo, halos, OPS_shape, k_chunk=k_chunk, dtype=np.float32)

        if h0_switch:
            with _Timer("Write h0"):
                write_grid_variable_from_vectors(g1, b, "h0", xv, yv, zv, nhalo, halos, OPS_shape, k_chunk=k_chunk, dtype=np.float32)

        h5f.flush()
        _log(f"Block {block_number} complete.")

    _log("All blocks written (vectors path).")



def write_grid_variable(group, block, name, data, halos, size):
    """
        Writes a grid array to an HDF5 group and sets OPS metadata.

        Parameters:
        - group: HDF5 group to write to
        - block: OpenSBLI SimulationBlock
        - name: e.g., "x0", "x1", "x2", "h0"
        - data: padded NumPy array
        - halos: [[-nhalo, -nhalo, -nhalo], [nhalo, nhalo, nhalo]]
        - size: [nx, ny, nz] (no halos)
    """
    # Create coordinates
    # dset_name = np.string_(block.location_dataset(name).base)
    dset_name = block.location_dataset(name).base

    # dset = group.create_dataset(dset_name, data=data)
    dset = group.create_dataset('%s' % (dset_name), data=data)


    set_hdf5_metadata(dset, halos=halos, npoints=size, block=block)

    del data
    gc.collect()





def write_blocks_to_hdf5(block_data, nblocks, h5f, nhalo=5, h0_switch=False):
    _log(f"Writing data to hdf5 for {nblocks} blocks")

    for block_number in range(nblocks):
        grid = block_data[block_number]
        ndim = 3 if 'z' in grid else 2

        _log(f"--- Block {block_number} (ndim={ndim}) ---")

        with _Timer("Create group + attributes"):
            b = SimulationBlock(ndim, block_number=block_number)
            g1 = h5f.create_group(b.blockname)
            apply_group_attributes(g1, b)

        halo = [[-nhalo] * ndim, [nhalo] * ndim]
        pad = extend_halo_3D if ndim == 3 else extend_halo_2D
        OPS_shape = list(grid['x'].shape)[::-1]



        # x
        with _Timer("Pad x"):
            full_x = pad(grid['x'], nhalo)
        with _Timer("Write x0"):
            write_grid_variable(g1, b, "x0", full_x, halos=halo, size=OPS_shape)
            del full_x
            gc.collect()


        # y
        with _Timer("Pad y"):
            full_y = pad(grid['y'], nhalo)
        with _Timer("Write x1"):
            write_grid_variable(g1, b, "x1", full_y, halos=halo, size=OPS_shape)
            del full_y
            gc.collect()


        # z
        if ndim == 3:
            with _Timer("Pad z"):
                full_z = pad(grid['z'], nhalo)
            with _Timer("Write x2"):
                write_grid_variable(g1, b, "x2", full_z, halos=halo, size=OPS_shape)
                del full_z
                gc.collect()

        if h0_switch:
            with _Timer("Pad h0"):
                full_h = pad(grid['h'], nhalo)
            with _Timer("Write h0"):
                write_grid_variable(g1, b, "h0", full_h, halos=halo, size=OPS_shape)
                del full_h
                gc.collect()

        _log(f"Block {block_number} complete.")
        h5f.flush()  # makes progress visible on disk (and can help diagnose I/O stalls)

    _log("All blocks written.")



def compute_h0(y):
    """
        Compute h0 from a 3D y-array.
        Collapses the wall-normal direction and broadcasts to full shape.
    """
    ndim = len(y.shape)
    if ndim == 3:
        h0_min = np.min(y, axis=1)  # collapse y-dir (axis 1)
        h0_broadcast = np.broadcast_to(h0_min[:, np.newaxis, :], y.shape)
    elif ndim == 2:
        h0_min = np.min(y, axis=0)  # collapse y-dir (axis 0)
        h0_broadcast = np.broadcast_to(h0_min[ np.newaxis, :], y.shape)
    return h0_broadcast





def read_blocks_from_arr(x_B, y_B, z_B):
    """ reads in the blocks from .dats"""
    block_data = {}
    nblocks = len(x_B)
    for i in range(nblocks):

        grid = {'x': x_B[i], 'y': y_B[i], 'z': z_B[i]}
        grid['h'] = compute_h0(grid['y'])

        block_data[i] = grid

    return block_data






# ----------------------------------------------------------------------------------------------------------------------------------
# check h5 file created correctly, visualise points
# ----------------------------------------------------------------------------------------------------------------------------------


def visualise_grid_from_hdf5(fname, nblocks, nhalo, h0_switch=True, step3d=10):
    print(f"plotting hdf5 grid file")


    with h5py.File(fname, 'r') as f:
        # Print datasets
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\n{name}:  {obj.shape}")

        print("Grid HDF5 contains:")
        f.visititems(print_datasets)

        # Probe first block to determine dimensionality
        probe_path = "/opensbliblock00/x0_B0"  # adjust if your naming differs
        dset = f[probe_path]
        ndim = dset.ndim

        if ndim == 2:
            plt.figure(figsize=(10,6))
            for i in range(nblocks):
                x_path = f"/opensbliblock0{i}/x0_B{i}"
                y_path = f"/opensbliblock0{i}/x1_B{i}"
                
                
                x0 = f[x_path]
                x1 = f[y_path]
                Ny, Nx = x0.shape

                x = x0[nhalo:Ny-nhalo, nhalo:Nx-nhalo]
                y = x1[nhalo:Ny-nhalo, nhalo:Nx-nhalo]

                plt.scatter(x.ravel(), y.ravel(), s=0.5)

            plt.axis('equal')
            plt.title("2D Grid")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()
            return

        elif ndim == 3:
            x_B, y_B, z_B, h0_B = {}, {}, {}, {}
            for i in range(nblocks):
                x_path = f"/opensbliblock0{i}/x0_B{i}"
                y_path = f"/opensbliblock0{i}/x1_B{i}"
                z_path = f"/opensbliblock0{i}/x2_B{i}"

                N2, N1, N0 = f[f'/opensbliblock0{i}/x0_B{i}'].shape
                slc = (slice(nhalo, N2 - nhalo), slice(nhalo, N1 - nhalo), slice(nhalo, N0 - nhalo))

                x_B[i] = f[x_path][slc]
                y_B[i] = f[y_path][slc]
                z_B[i] = f[z_path][slc]

                if h0_switch:
                    h0_path = f"/opensbliblock0{i}/h0_B{i}"
                    if h0_path in f:
                        h0_B[i] = f[h0_path][slc]        

            # 2D slice plot (x-y at z-index 0)
            plt.figure(figsize=(10, 6))
            for i in range(nblocks):
                plt.scatter(x_B[i][:, :, 0].ravel(), y_B[i][:, :, 0].ravel(), s=0.5, marker='.')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Generated Grid Points (slice)')
            plt.axis('equal')
            plt.grid(True)
            plt.show()




            # 3D subsampled plot
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(nblocks):
                xs = x_B[i][::step3d, ::step3d, ::step3d].ravel()
                ys = y_B[i][::step3d, ::step3d, ::step3d].ravel()
                zs = z_B[i][::step3d, ::step3d, ::step3d].ravel()

                # you had (x, z, y); keep if that's your preferred orientation
                ax.scatter(xs, zs, ys, s=0.5)

            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')
            ax.set_title('3D Subsampled Grid')
            plt.show()

        else:
            raise ValueError(f"Unsupported dataset dimensionality: {ndim}")


# ---------------------------------------------------------------------------------------------------------------------



def arrays_to_HDF5(casename, x_B, y_B, z_B, plot_hdf5=True):
    os.makedirs(GRID_DIR, exist_ok=True)                  # ensure it exists

    fname = os.path.join(GRID_DIR, f"{casename.lower()}.h5")

    nblocks = len(x_B)



    with h5py.File(fname, 'w') as h5f:

        print(f"Reading data for {nblocks} blocks")
        block_data = read_blocks_from_arr(x_B, y_B, z_B)


        write_blocks_to_hdf5(block_data, nblocks, h5f,  h0_switch=True)


    if plot_hdf5:
        print(f"Plotting HDF5")
        visualise_grid_from_hdf5(fname, nblocks, 5, h0_switch)

    return
    

def vectors_to_HDF5(casename, x_vec, y_vec, z_vec, plot_hdf5=True, nhalo=5):
    fname = os.path.join(GRID_DIR, f"{casename.lower()}.h5")
    nblocks = len(x_vec)

    with h5py.File(fname, 'w') as h5f:
        print(f"Writing vectors for {nblocks} blocks (NO 3D numpy arrays)...", flush=True)

        # THIS is where vectors become 3D arrays in the HDF5 file:
        write_blocks_to_hdf5_from_vectors(x_vec, y_vec, z_vec, nblocks, h5f, nhalo=nhalo, h0_switch=True, k_chunk=16)

    if plot_hdf5:
        print("Plotting HDF5", flush=True)
        visualise_grid_from_hdf5(fname, nblocks, nhalo, h0_switch=True)

    return
