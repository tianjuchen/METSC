import numpy as np
from PIL import Image
import nibabel as nib
import sys

def generate_mask_from_png(png_path, threshold, output_nii_path, target_shape=(256, 256)):
    """
    从 PNG 图像生成掩码并保存为 NIfTI 格式
    :param png_path: 输入 PNG 图像的路径
    :param threshold: 用于生成掩码的像素阈值
    :param output_nii_path: 输出 NIfTI 掩码文件的路径
    :param target_shape: 目标图像形状，默认为 (256, 256)
    """
    try:
        # 打开 PNG 图像
        img = Image.open(png_path)

        # 调整图像大小
        img = img.resize(target_shape, Image.BICUBIC)

        # 将图像转换为 numpy 数组
        img_array = np.array(img)
        
        # 如果是彩色图像，转换为灰度图像
        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            
        # 生成二值掩码
        mask = np.where(gray_img > threshold, 1, 0).astype(np.uint8)

        # 创建 NIfTI 图像对象
        # 假设仿射矩阵为单位矩阵，对于大多数情况简单处理
        affine = np.eye(4)
        mask_nii = nib.Nifti1Image(mask, affine)
        
        # 保存 NIfTI 图像
        nib.save(mask_nii, output_nii_path)
        print(f"Mask saved to {output_nii_path}")
    except FileNotFoundError:
        print(f"Error: The file {png_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 输入 PNG 图像的路径
    input_png_path = "./image/550-500h-source.png"
    # 输出 NIfTI 掩码文件的路径
    output_nii_path = "./image/mask.nii"
    # 阈值，可根据实际情况调整
    threshold = 128
    # 目标图像形状
    target_shape = (7, 9)

    generate_mask_from_png(input_png_path, threshold, output_nii_path, target_shape=target_shape)