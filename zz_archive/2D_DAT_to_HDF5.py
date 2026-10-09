
# Converts DAT grid coords to HDF5 format. For reading into OpenSBLI
# need to check if it reads in z coords from the dat unlike the old version
# 
# and it adds a column for H0 (the min height of the block)
#
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

casename = "florida_2d"
nblocks = 4
input_files = [os.path.join(GRID_DIR, f"{casename}_B{i}.dat") for i in range(nblocks)]

# Number of halo points to add on each side of each direction (default 5)
nhalo = 5
# number of dimensions
# Output grid file name
fname = os.path.join(GRID_DIR, f"{casename.lower()}.h5")

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


def fill_halo_coordinates(block_data, block_number):
    """ Function to fill the halo points in the x, y, z coordinates. """
    # Unpack the block data
    x = block_data[block_number]['x']
    y = block_data[block_number]['y']
    h = block_data[block_number]['h']
    # print("fill halo h shape: ", h.shape)
    
    
    ny, nx = x.shape  # 2D shapes
    shape = [ny + 2 * nhalo, nx + 2 * nhalo]
    # print("shape: ", shape)


    # Shape without halo points (Nx, Ny, Nz) required for OPS attribute, without halos
    OPS_shape = list(x.shape)[::-1]

    # Make an OpenSBLI block
    b = SimulationBlock(2, block_number=block_number)  # 2D now
    g1 = h5f.create_group(b.blockname)
    halo = [[-nhalo, -nhalo], [nhalo, nhalo]]  # 2D halo
    apply_group_attributes(g1, b)
    
    # Initialize arrays with halos
    full_x = np.zeros(shape)
    # Fill interior data

    # Fill interior data
    full_x[nhalo:nhalo + ny, nhalo:nhalo + nx] = x

    # Fill top and bottom halo rows
    for j in range(ny + 2 * nhalo):
        if j < nhalo:  # Top
            full_x[j, nhalo:nhalo + nx] = full_x[nhalo, nhalo:nhalo + nx] - (
                full_x[nhalo + 1, nhalo:nhalo + nx] - full_x[nhalo, nhalo:nhalo + nx]
            )
        elif j >= ny + nhalo:  # Bottom
            full_x[j, nhalo:nhalo + nx] = full_x[ny + nhalo - 1, nhalo:nhalo + nx] + (
                full_x[ny + nhalo - 1, nhalo:nhalo + nx] - full_x[ny + nhalo - 2, nhalo:nhalo + nx]
            )

    # Left and right halo columns (incrementally)
    left_diff = full_x[nhalo:nhalo + ny, nhalo + 1] - full_x[nhalo:nhalo + ny, nhalo]
    right_diff = full_x[nhalo:nhalo + ny, nx + nhalo - 1] - full_x[nhalo:nhalo + ny, nx + nhalo - 2]

    for i in range(nx + 2 * nhalo):
        if i < nhalo:  # Left
            full_x[nhalo:nhalo + ny, i] = full_x[nhalo:nhalo + ny, nhalo] - left_diff * (nhalo - i)
        elif i >= nx + nhalo:  # Right
            full_x[nhalo:nhalo + ny, i] = full_x[nhalo:nhalo + ny, nx + nhalo - 1] + right_diff * (i - (nx + nhalo - 1))

    # Write to HDF5
    block_dset_name = str(b.location_dataset("x0").base)
    print("OpenSBLI block shape without halo points: %s" % OPS_shape)

    dset = g1.create_dataset(block_dset_name, data=full_x)
    set_hdf5_metadata(dset, halos=halo, npoints=OPS_shape, block=b)
    print('x0 added to data.h5')
    del full_x
    gc.collect()

    # Initialize full_y with halos
    shape = [ny + 2 * nhalo, nx + 2 * nhalo]
    full_y = np.zeros(shape)

    # Fill interior
    full_y[nhalo:nhalo + ny, nhalo:nhalo + nx] = y

    # Fill top and bottom halo rows
    for j in range(ny + 2 * nhalo):
        if j < nhalo:  # Top
            full_y[j, nhalo:nhalo + nx] = full_y[nhalo, nhalo:nhalo + nx] - (
                full_y[nhalo + 1, nhalo:nhalo + nx] - full_y[nhalo, nhalo:nhalo + nx]
            )
        elif j >= ny + nhalo:  # Bottom
            full_y[j, nhalo:nhalo + nx] = full_y[ny + nhalo - 1, nhalo:nhalo + nx] + (
                full_y[ny + nhalo - 1, nhalo:nhalo + nx] - full_y[ny + nhalo - 2, nhalo:nhalo + nx]
            )

    # Fill left and right halo columns incrementally
    left_diff = full_y[nhalo:nhalo + ny, nhalo + 1] - full_y[nhalo:nhalo + ny, nhalo]
    right_diff = full_y[nhalo:nhalo + ny, nx + nhalo - 1] - full_y[nhalo:nhalo + ny, nx + nhalo - 2]

    for i in range(nx + 2 * nhalo):
        if i < nhalo:  # Left
            full_y[nhalo:nhalo + ny, i] = full_y[nhalo:nhalo + ny, nhalo] - left_diff * (nhalo - i)
        elif i >= nx + nhalo:  # Right
            full_y[nhalo:nhalo + ny, i] = full_y[nhalo:nhalo + ny, nx + nhalo - 1] + right_diff * (i - (nx + nhalo - 1))

    # Create y dataset
    block_dset_name = str(b.location_dataset("x1").base)
    dset = g1.create_dataset(block_dset_name, data=full_y)
    set_hdf5_metadata(dset, halos=halo, npoints=OPS_shape, block=b)
    print('x1 added to data.h5')

    # Clean up
    del full_y
    gc.collect()

    
    if h0_switch:
        full_h = np.zeros(shape)
        full_h[nhalo:nhalo + ny, nhalo:nhalo + nx] = h
        print("full_h shape: ", full_h.shape)

 
        for j in range(ny + 2 * nhalo):
            # Left (j < nhalo)
            if j < nhalo:
                full_h[j, nhalo:nhalo+nx] = full_h[nhalo+1, nhalo:nhalo+nx] 
                # full_h[nhalo:nhalo+nz, j, nhalo:nhalo+nx] = (nhalo-j)*full_h[nhalo:nhalo+nz, nhalo+1, nhalo:nhalo+nx] 
            # Right (j >= ny + nhalo)
            if j >= ny + nhalo:
                full_h[j, nhalo:nhalo+nx] = full_h[ny + nhalo - 1, nhalo:nhalo+nx] 
                # full_h[nhalo:nhalo+nz, j, nhalo:nhalo+nx] = full_h[nhalo:nhalo+nz, ny + nhalo - 1, nhalo:nhalo+nx] + (j-nhalo-ny+1)*(full_h[nhalo:nhalo+nz, ny + nhalo - 1, nhalo:nhalo+nx] - full_h[nhalo:nhalo+nz, ny + nhalo - 2, nhalo:nhalo+nx])

        for i in range(nx + 2 * nhalo):
            # Front (i < nhalo)
            if i < nhalo:
                full_h[nhalo:nhalo+ny, i] = full_h[nhalo:nhalo+ny, nhalo] 
            # Back (i >= nx + nhalo)
            if i >= nx + nhalo:
                full_h[nhalo:nhalo+ny, i] = full_h[nhalo:nhalo+ny, nx + nhalo - 1] 

        # Create h coordinates
        block_dset_name = b.location_dataset("h0").base
        dset = g1.create_dataset('%s' % (block_dset_name), data=full_h)
        set_hdf5_metadata(dset, halos=halo, npoints=OPS_shape, block=b)
        print('h0 added to data.h5')
        del full_h
        gc.collect()


    return 




def read_blocks(block_data):
    total_grid_points = 0
    for block_number, block in enumerate(input_files):
        print("\n\nReading from %s." % block)
        with open(block) as f:
            nx, ny = map(int, f.readline().split())
        print("Nx, Ny from the input file for block %d:" % block_number, nx, ny)
        
        # Read the data
        x, y= np.loadtxt(block, skiprows=1, unpack=True)



        x = x.reshape(nx, ny)
        y = y.reshape(nx, ny)
        
        
        # Transpose to match desired order
        x = x.transpose((1, 0))
        y = y.transpose((1, 0))
        
        


        h = y
        h0 = np.min(y, axis=0)  # collapse y-direction
        print("h0 shape:", h0.shape)
        h = np.broadcast_to(h0[np.newaxis, :], y.shape)  
        
        print("h shape after tiling:", h.shape)




        # h = np.tile(np.min(y, axis=0), (y.shape[0], 1))


        shape = list(x.shape)
        total = shape[0] * shape[1]
        print("Block %d has %e grid points." % (block_number, int(total)))
        total_grid_points += total
        print("Original 2D shape: %s" % shape)
        
        # Save this block data to the dictionary
        block_data[block_number] = {'x': x, 'y': y, 'h': h}
    
    # Finish
    print("Total grid points: %g" % total_grid_points)
    return

# ---------------------------------------------------------------------------------------------------------------------


block_data = {}


h5f = h5py.File(fname, 'w')


 
# Loop over all of the grid points
for block_number, block in enumerate(input_files):
    # Read all of the blocks at once
    if block_number == 0:
        print("Reading all of the x, y data for the %d blocks" % nblocks)
        read_blocks(block_data)
    # Extend the coordinates into the halo points to avoid issues with the metric calculations
    fill_halo_coordinates(block_data, block_number)

    x, y, h = block_data[block_number]['x'], block_data[block_number]['y'], block_data[block_number]['h'] 
    print("Length in x for block %d:" % block_number, abs(np.amin(x) - np.amax(x)))
    print("Length in y for block %d:" % block_number, abs(np.amin(y) - np.amax(y)))

 
h5f.close()




# ----------------------------------------------------------------------------------------------------------------------------------
# check h5 file created correctly, visualise points
# ----------------------------------------------------------------------------------------------------------------------------------

print("\n", "-" * 30, f"\n{fname} created. \n", "-" * 30)


# Load data from HDF5 file
with h5py.File(fname, 'r') as f:


    # Print all items in h5 file
    def print_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):  # Check if object is a dataset
            print(f"{name}:  {obj.shape}")
            print("-" * 30)

    # Visit all datasets and print their details
    f.visititems(print_datasets)


    y_B = {}
    x_B = {}
    h0_B = {}

    for i in range(nblocks):
        # Find the size of x, y, z
        (N1, N0) = f[f'/opensbliblock0{i}/x0_B{i}'].shape

        print(f'Block {i} Grid size with Halos in x, y: ', N0, N1)

        # slice off the halo values: x=0 at n=5, y=0 at n=5
        x_B[i]    = f[f'/opensbliblock0{i}/x0_B{i}'][nhalo:N1 - nhalo, nhalo:N0 - nhalo]             
        y_B[i]    = f[f'/opensbliblock0{i}/x1_B{i}'][nhalo:N1 - nhalo, nhalo:N0 - nhalo]

        if h0_switch:
            h0_B[i] = f[f'/opensbliblock0{i}/h0_B{i}'][nhalo:N1 - nhalo, nhalo:N0 - nhalo]





# Plot the scatter plot of the grid points
plt.figure(figsize=(10, 6))

for i in range(nblocks):
    
    plt.scatter(x_B[i][: , :], y_B[i][:, :], s=0.5, marker='.')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Grid Points 2D')
plt.axis('equal')  # Maintain equal scaling of axes
plt.grid(True)



if h0_switch:

    plt.figure(figsize=(8, 6))
    for i in range(nblocks):
        # Flatten the arrays for scatter plot
        x_flat = x_B[i].flatten()
        h0_flat = h0_B[i].flatten()

        plt.scatter(x_flat, h0_flat, s=1, alpha=0.5, marker='.', label=f'Block {i}')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$h_0$')
    plt.title(r'Scatter Plot of $h_0$ vs $x$')
    plt.legend()
    plt.grid(True)


plt.show()





# ---------------------------------------------------------------------------------------------------------------------
