import nibabel as nib
import numpy as np
from PIL import Image
import os

def convert_nii_to_jpg(nii_file, output_dir):
    """
    Converts a NIfTI file to a series of JPG images, one for each slice.

    Args:
        nii_file (str): Path to the input NIfTI file (.nii or .nii.gz).
        output_dir (str): Path to the directory where JPG images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        img = nib.load(nii_file)
        data = img.get_fdata()
    except Exception as e:
         raise Exception(f"Error loading NIfTI file: {e}")

    # Normalize the data to 0-255 range and convert to uint8
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    
    for i in range(data.shape[2]):  # Iterate through slices
        slice_data = data[:, :, i]
        image = Image.fromarray(slice_data)
        image_path = os.path.join(output_dir, f"slice_{i:03d}.jpg")
        image.save(image_path)


# Example usage:
# nii_file_path = "path/to/your/image.nii.gz"  # Replace with your NIfTI file path
# output_directory = "path/to/output/jpgs"  # Replace with your desired output directory

nii_file_path = input("NIFTI file path: " )
output_directory = input("Desired output directory: " )
convert_nii_to_jpg(nii_file_path, output_directory)