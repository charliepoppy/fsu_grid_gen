
import numpy as np
import os

# Resolve the path to the sister folder ("../grids")
BASE_DIR = os.path.dirname(__file__)                 # folder containing this script
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "grids"))

os.makedirs(SAVE_DIR, exist_ok=True)  # ensure it exists




def save_block_to_dat_2D(x, y, filename):
    #  must be fed in shape nx, ny so might need to transpose first
    data = np.column_stack((x.ravel(), y.ravel()))

    nx, ny = x.shape
    
    header = f"{nx} {ny}"
    filepath = os.path.join(SAVE_DIR, filename)

    print(f"Saving {filename} ... shape: {data.shape}")
    np.savetxt(filepath, data, header=header, comments='')
    return


def save_block_to_dat_3D(x, y, z, filename):
    data = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    nx, ny, nz = x.shape

    header = f"{nx} {ny} {nz}"
    filepath = os.path.join(SAVE_DIR, filename)

    print(f"Saving {filename} ... shape: {data.shape}")
    np.savetxt(filepath, data, header=header, comments='')
    print("saved.")
    return
