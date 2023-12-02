import torch
import numpy as np

def combine_parts(original_image, not_attacked_part, corrected_part):
    # 将 PyTorch 张量转换为 NumPy 数组
    original_array = original_image.detach().cpu().numpy()
    not_attacked_array = not_attacked_part.detach().cpu().numpy()
    corrected_array = corrected_part.detach().cpu().numpy()

    # 确保所有数组都是3维的（C, H, W）
    if original_array.ndim == 4:
        original_array = original_array.squeeze(0)
    if not_attacked_array.ndim == 4:
        not_attacked_array = not_attacked_array.squeeze(0)
    if corrected_array.ndim == 4:
        corrected_array = corrected_array.squeeze(0)

    # 打印形状以用于调试
    print("Original array shape:", original_array.shape)
    print("Not attacked array shape:", not_attacked_array.shape)
    print("Corrected array shape:", corrected_array.shape)

    # 确保所有数组的形状相同
    if original_array.shape != not_attacked_array.shape or original_array.shape != corrected_array.shape:
        raise ValueError("All input arrays must have the same shape.")

    # 直接相加未受攻击部分和校正部分，并减去重叠部分
    combined_array = not_attacked_array + corrected_array - original_array

    # 对结果进行裁剪以确保值在合法范围内
    combined_array = np.clip(combined_array, 0, 1)

    # 将合并后的数组转换回 PyTorch 张量
    combined_tensor = torch.from_numpy(combined_array)

    return combined_tensor

