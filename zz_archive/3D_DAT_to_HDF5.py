"""
Converts DAT grid coords to HDF5 format. For reading into OpenSBLI
need to check if it reads in z coords from the dat unlike the old version

and it adds a column for H0 (the min height of the block)
"""

# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import h5py
import matplotlib.pyplot as plt
from opensbli import *
import gc

# ---------- paths ----------
HERE = os.path.dirname(__file__)                       # directory of this script
GRID_DIR = os.path.abspath(os.path.join(HERE, "grids"))  # ./grids next to script
os.makedirs(GRID_DIR, exist_ok=True)                  # ensure it exists

# Inputs
# ---------------------------------------------------------------------------------------------------------------------

# Specify the input grid files
casename = "florida_3d"

nblocks = 4
h0_switch = 1
# ---------------------------------------------------------------------------------------------------------------------

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


def output_hdf5(array, array_name, halos, npoints, block):
    """ Creates an HDF5 file for reading in data to a simulation, 
    sets the metadata required by the OPS library. """
    if not isinstance(array, list):
        array = [array]
    if not isinstance(array_name, list):
        array_name = [array_name]
    assert len(array) == len(array_name)
    with h5py.File('data.h5', 'w') as hf:
        # Create a group
        if isinstance(block, MultiBlock):
            all_blocks = block.blocks
        else:
            all_blocks = [block]
        for b in all_blocks:
            g1 = hf.create_group(b.blockname)
            # Loop over all the dataset inputs and write to the hdf5 file
            for ar, name in zip(array, array_name):
                g1.attrs.create("dims", [b.ndim], dtype="int32")
                g1.attrs.create("ops_type", u"ops_block", dtype="S9")
                g1.attrs.create("index", [b.blocknumber], dtype="int32")
                block_dset_name = b.location_dataset(name).base
                dset = g1.create_dataset('%s' % (block_dset_name), data=ar)
                set_hdf5_metadata(dset, halos, npoints, b)
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
    print(f'{name} added to data.h5')

    del data
    gc.collect()



def write_block_to_hdf5(block_data, block_number, h5f, nhalo=5, h0_switch=False):
    x, y, z, h = block_data[block_number]['x'], block_data[block_number]['y'], block_data[block_number]['z'], block_data[block_number]['h']
    print("Block %d: x shape: %s, y shape: %s, z shape: %s, h shape: %s" % (block_number, x.shape, y.shape, z.shape, h.shape))



    # Shape without halo points (Nx, Ny, Nz) required for OPS attribute, without halos
    OPS_shape = list(x.shape)[::-1]
    # Make an OpenSBLI block
    b = SimulationBlock(3, block_number=block_number)
    g1 = h5f.create_group(b.blockname)
    halo = [[-nhalo, -nhalo, -nhalo], [nhalo, nhalo, nhalo]]
    apply_group_attributes(g1, b)
    


    # Initialize arrays with halos
    full_x = extend_halo_3D(x, nhalo)
    full_y = extend_halo_3D(y, nhalo)
    full_z = extend_halo_3D(z, nhalo)

    if h0_switch:
        full_h = extend_halo_3D(h, nhalo)



    write_grid_variable(g1, b, "x1", full_y, halos=halo, size=OPS_shape)
    write_grid_variable(g1, b, "x2", full_z, halos=halo, size=OPS_shape)
    write_grid_variable(g1, b, "x0", full_x, halos=halo, size=OPS_shape)

    if h0_switch:
        write_grid_variable(g1, b, "h0", full_h, halos=halo, size=OPS_shape)




    return 


def read_dat_grid(filepath):
    """Reads a .dat grid file and returns x, y, z arrays in [z, y, x] order."""
    with open(filepath) as f:
        nx, ny, nz = map(int, f.readline().split())
    x, y, z = np.loadtxt(filepath, skiprows=1, unpack=True)
    x = x.reshape(nx, ny, nz).transpose((2, 1, 0))
    y = y.reshape(nx, ny, nz).transpose((2, 1, 0))
    z = z.reshape(nx, ny, nz).transpose((2, 1, 0))
    return x, y, z

def compute_h0(y):
    """
    Compute h0 from a 3D y-array.
    Collapses the wall-normal direction and broadcasts to full shape.
    """
    h0_min = np.min(y, axis=1)  # collapse y-dir (axis 1)
    h0_broadcast = np.broadcast_to(h0_min[:, np.newaxis, :], y.shape)
    return h0_broadcast





def read_blocks(block_data, input_files):
    total_grid_points = 0
    for block_number, block in enumerate(input_files):
        print("\n\nReading from %s." % block)
        x, y, z = read_dat_grid(block)
        h = compute_h0(y)



        print("h shape after tiling:", h.shape)

        shape = list(x.shape)
        total = shape[0] * shape[1] * shape[2]
        print("Block %d has %e grid points." % (block_number, int(total)))
        total_grid_points += total
        print("Original 3D shape: %s" % shape)
        
        # Save this block data to the dictionary
        block_data[block_number] = {'x': x, 'y': y, 'z': z, 'h': h}
    
    # Finish
    print("Total grid points: %g" % total_grid_points)
    return





# ----------------------------------------------------------------------------------------------------------------------------------
# check h5 file created correctly, visualise points
# ----------------------------------------------------------------------------------------------------------------------------------


def visualise_grid_from_hdf5(fname, nhalo, h0_switch=True):
    with h5py.File(fname, 'r') as f:
        x_B, y_B, z_B, h0_B = {}, {}, {}, {}


        # Print all items in h5 file
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):  # Check if object is a dataset
                print(f"\n{name}:  {obj.shape}")
                print("-" * 30)
        # Visit all datasets and print their details
        f.visititems(print_datasets)



        for i in range(nblocks):
            # Find the size of x, y, z
            (N2, N1, N0) = f[f'/opensbliblock0{i}/x0_B{i}'].shape
            x_B[i]    = f[f'/opensbliblock0{i}/x0_B{i}'][nhalo:N2 - nhalo, nhalo:N1 - nhalo, nhalo:N0 - nhalo]             
            y_B[i]    = f[f'/opensbliblock0{i}/x1_B{i}'][nhalo:N2 - nhalo, nhalo:N1 - nhalo, nhalo:N0 - nhalo]
            z_B[i]    = f[f'/opensbliblock0{i}/x2_B{i}'][nhalo:N2 - nhalo, nhalo:N1 - nhalo, nhalo:N0 - nhalo]
            if h0_switch:
                h0_B[i] = f[f'/opensbliblock0{i}/h0_B{i}'][nhalo:N2 - nhalo, nhalo:N1 - nhalo, nhalo:N0 - nhalo]
    

        # 2D plot
        plt.figure(figsize=(10, 6))
        for i in range(nblocks):
            plt.scatter(x_B[i][0, :, :], y_B[i][0, :, :], s=0.5, marker='.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Generated Grid Points 2D')
        plt.axis('equal')
        plt.grid(True)



        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        step = 10
        for i in range(nblocks):
            ax.scatter(x_B[i][::step, ::step, ::step].flatten(),
                       z_B[i][::step, ::step, ::step].flatten(),
                       y_B[i][::step, ::step, ::step].flatten(), s=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title('3D Subsampled Grid')



        if h0_switch:
            plt.figure(figsize=(8, 6))
            for i in range(nblocks):
                plt.scatter(x_B[i][0, 0, :], h0_B[i][0, 0, :], s=1, alpha=0.5)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$h_0$')
            plt.title(r'Scatter Plot of $h_0$ vs $x$')
            plt.grid(True)

        plt.show()


# ---------------------------------------------------------------------------------------------------------------------




def main():
    block_data = {}
    fname = os.path.join(GRID_DIR, f"{casename.lower()}.h5")
    input_files = [os.path.join(GRID_DIR, f"{casename}_B{i}.dat") for i in range(nblocks)]

    with h5py.File(fname, 'w') as h5f:
        print(f"Reading all of the x, y, z data for {nblocks} blocks")
        read_blocks(block_data, input_files)




        # Loop over all of the grid points
        for block_number, _ in enumerate(input_files):

            write_block_to_hdf5(block_data, block_number, h5f,  h0_switch=True)

            x = block_data[block_number]['x']
            y = block_data[block_number]['y']
            z = block_data[block_number]['z']      
            print("Length in x for block %d:" % block_number, abs(np.amin(x) - np.amax(x)))
            print("Length in y for block %d:" % block_number, abs(np.amin(y) - np.amax(y)))
            print("Length in z for block %d:" % block_number, abs(np.amin(z) - np.amax(z)))



    visualise_grid_from_hdf5(fname, h0_switch)



    return




if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------
