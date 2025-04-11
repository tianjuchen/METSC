import nibabel as nib
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def normalize_slice(slice_data):
    """Normalize slice data for better visualization"""
    # Remove outliers using percentiles
    p1, p99 = np.percentile(slice_data, 1), np.percentile(slice_data, 99)
    slice_data = np.clip(slice_data, p1, p99)
    
    # Normalize to 0-1 range
    if np.max(slice_data) > np.min(slice_data):
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
    return slice_data

def convert_nii_to_jpg(nii_file, output_dir):
    """
    Enhanced NIfTI to JPG conversion with better visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        img = nib.load(nii_file)
        data = img.get_fdata()
        print(f"Loaded NIfTI file: {nii_file}, data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return

    # Create a montage of slices for better visualization
    if len(data.shape) == 2:
        # Handle 2D case
        slice_data = normalize_slice(data)
        plt.imshow(slice_data, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        output_path = os.path.join(output_dir, 'slice.jpg')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"Saved 2D slice to {output_path}")
        return
    elif len(data.shape) == 3:
        # Handle 3D case
        num_slices = data.shape[2]
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        
        # Flatten axes array for easy iteration
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i in range(num_slices):
            ax = axes_flat[i]
            slice_data = data[:, :, i]
            slice_data = normalize_slice(slice_data)
            ax.imshow(slice_data, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        
        # Remove empty subplots
        for j in range(num_slices, len(axes_flat)):
            axes_flat[j].axis('off')
        
        # Save the figure
        output_path = os.path.join(output_dir, 'montage.jpg')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"Saved montage to {output_path}")

        # Save individual slices
        for i in range(num_slices):
            slice_data = data[:, :, i]
            slice_data = normalize_slice(slice_data)
            
            # Convert to 8-bit and save
            slice_8bit = (slice_data * 255).astype(np.uint8)
            image = Image.fromarray(slice_8bit)
            image_path = os.path.join(output_dir, f"slice_{i:03d}.jpg")
            try:
                image.save(image_path)
                print(f"Saved slice {i} to {image_path}")
            except Exception as e:
                print(f"Error saving slice {i}: {e}")
    else:
        print(f"Unsupported data dimensionality: {len(data.shape)}")

if __name__ == "__main__":
    nii_file_path = input("NIFTI file path: ")
    output_directory = input("Desired output directory: ")
    convert_nii_to_jpg(nii_file_path, output_directory)