from scipy.io import loadmat
import h5py
import os

def safe_load_mat(filepath):
    """Attempt to load .mat file with multiple methods"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError("File is empty (0 bytes)")
    
    try:
        # Try standard MATLAB format
        return loadmat(filepath)
    except Exception as e1:
        try:
            # Try HDF5 format (MATLAB v7.3)
            with h5py.File(filepath, 'r') as f:
                return {k: f[k][()] for k in f.keys()}
        except Exception as e2:
            raise ValueError(
                f"Failed to load .mat file:\n"
                f"Standard format error: {str(e1)}\n"
                f"HDF5 format error: {str(e2)}"
            )
# Usage:
try:
    data = safe_load_mat('./image/data.mat')
    print("Loaded variables:", [k for k in data if not k.startswith('__')])
except Exception as e:
    print("Error:", e)