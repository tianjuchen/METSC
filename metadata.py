import numpy as np

# Generate sample grain size distribution (64 bins)
grain_size_dist = np.random.rand(64)
grain_size_dist = grain_size_dist / grain_size_dist.sum()  # Normalize to probabilities

# Generate sample yield strength
yield_strength = np.random.rand(1) * 100  # Random value between 0 and 100 MPa

# Save to .npz file
path = "/mnt/c/Users/Admin/Desktop/METSC/nanomaterials/"
np.savez(
    path + "image_001.npz", grain_size_dist=grain_size_dist, yield_strength=yield_strength
)
