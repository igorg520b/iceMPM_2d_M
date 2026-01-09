
import h5py
import sys

def print_structure(name, obj):
    print(name)

try:
    path = "/home/s2/Projects-CUDA/iceMPM_2d_M/_input_data/nares_strait_with_wind/data.nc"
    print(f"Opening {path}")
    with h5py.File(path, 'r') as f:
        print("Keys:", list(f.keys()))
        f.visititems(print_structure)
except Exception as e:
    print(f"Error: {e}")
