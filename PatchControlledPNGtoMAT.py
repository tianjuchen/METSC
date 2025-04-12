import numpy as np
import h5py
import os
from PIL import Image
import argparse

class PNGtoMATConverter:
    def __init__(self, input_channels=60, output_channels=3, patch_size=3, max_patches=None):
        """
        Initialize converter to match Mydataset requirements

        Args:
            input_channels: int (default=60) - Number of input channels
            output_channels: int (default=3) - Number of output channels
            patch_size: int (default=3) - Size of image patches
            max_patches: int or None (default=None) - Maximum number of patches to extract
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.max_patches = max_patches

    def _process_image(self, img_path):
        """Process image and extract 3×3 patches"""
        img = Image.open(img_path)
        img_array = np.array(img)

        # Convert to float32 and normalize
        img_array = img_array.astype(np.float32) / 255.0

        # Handle channel dimension
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array]*self.input_channels, axis=-1)
        elif len(img_array.shape) == 3:  # Color
            if img_array.shape[2] < self.input_channels:
                repeats = (self.input_channels // img_array.shape[2]) + 1
                img_array = np.tile(img_array, (1, 1, repeats))[:, :, :self.input_channels]

        # Reshape to (H, W, C) and extract patches
        patches = []
        patch_count = 0
        for i in range(0, img_array.shape[0] - self.patch_size + 1, self.patch_size):
            for j in range(0, img_array.shape[1] - self.patch_size + 1, self.patch_size):
                if self.max_patches is not None and patch_count >= self.max_patches:
                    break
                patch = img_array[i:i+self.patch_size, j:j+self.patch_size, :]
                patches.append(patch)
                patch_count += 1
            if self.max_patches is not None and patch_count >= self.max_patches:
                break

        # Convert to (N, C, H, W)
        patches = np.stack(patches)
        patches = np.moveaxis(patches, -1, 1)  # (N, H, W, C) -> (N, C, H, W)

        # Create dummy output (center pixel values)
        outputs = np.random.rand(len(patches), self.output_channels)

        return patches, outputs

    def convert(self, png_path, output_path):
        """
        Convert PNG to hdf5 .mat file matching Mydataset format

        Args:
            png_path: str - Input PNG path
            output_path: str - Output .mat path
        """
        input_patches, output_values = self._process_image(png_path)
        num_patches = len(input_patches)

        # Create combined array (N, 63, 3, 3)
        combined_data = np.zeros((num_patches, 
                                self.input_channels + self.output_channels,
                                self.patch_size, 
                                self.patch_size))

        # Assign input patches
        combined_data[:, :self.input_channels] = input_patches

        # Assign output values to center pixels
        combined_data[:, self.input_channels:, 1, 1] = output_values

        # Save as hdf5 file
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=combined_data)

        print(f"Converted {png_path} to {output_path}")
        print(f"Created {num_patches} patches of size {self.patch_size}x{self.patch_size}")

    @staticmethod
    def batch_convert(input_dir, output_dir, file_extension='.png', max_patches=None):
        """Batch convert PNGs to .mat files"""
        converter = PNGtoMATConverter(max_patches=max_patches)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith(file_extension):
                png_path = os.path.join(input_dir, filename)
                mat_filename = os.path.splitext(filename)[0] + '.mat'
                output_path = os.path.join(output_dir, mat_filename)
                converter.convert(png_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PNGs to Mydataset-compatible .mat files')
    parser.add_argument('--input', type=str, default='./image/550-500h-source.png', help='Input PNG file/directory')
    parser.add_argument('--output', type=str, default='./image/data.mat', help='Output .mat file/directory')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--max_patches', type=int, default=None, help='Maximum number of patches to extract')

    args = parser.parse_args()

    if args.batch:
        PNGtoMATConverter.batch_convert(args.input, args.output, max_patches=args.max_patches)
    else:
        ic, oc, patch = 60, 3, 3
        converter = PNGtoMATConverter(input_channels = ic, output_channels = oc, patch_size = patch, max_patches=args.max_patches)
        converter.convert(args.input, args.output)
        
        
    
    #running command: python PNGtoMATConverter.py --input ./image/ --output ./output/ --batch --max_patches 25230