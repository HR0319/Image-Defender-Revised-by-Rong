import torch
import torch.nn as nn
from torch.autograd import Variable

def separate_attack(model, original_image, perturbed_image, k=1):
    # 创建一个新的叶子张量，复制 perturbed_image 的数据
    perturbed_image_clone = perturbed_image.clone().detach()
    perturbed_image_clone.requires_grad = True

    # 获取模型的输出
    original_output = model(original_image)
    perturbed_output = model(perturbed_image_clone)

    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(perturbed_output, torch.argmax(original_output, dim=1))

    # 计算梯度
    loss.backward()

    # 检查梯度计算
    if perturbed_image_clone.grad is None:
        raise ValueError("Gradient is not computed.")

    # 计算原始图像和受攻击图像之间的差异
    difference = torch.abs(perturbed_image_clone.grad.data)

    # 计算动态阈值
    mean_diff = torch.mean(difference)
    std_diff = torch.std(difference)
    dynamic_threshold = mean_diff + k * std_diff

    # 创建二值掩码
    mask = difference > dynamic_threshold

    # 分离受攻击的部分和未受攻击的部分
    attacked_part = perturbed_image.clone()
    unattacked_part = original_image.clone()

    # 从原始图像中提取被攻击的部分
    attacked_part[mask] = original_image[mask]

    # 从受攻击的图像中提取未被攻击的部分
    unattacked_part[~mask] = perturbed_image[~mask]

    return attacked_part, unattacked_part
