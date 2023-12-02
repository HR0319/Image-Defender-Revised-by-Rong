import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from adversarial_attack import adversarial_attack
from separate_attack import separate_attack
from correct_attack import correct_attack
from combine_parts import combine_parts
from torchvision.models import resnet18
from test import test
import tkinter as tk
from PIL import Image, ImageTk

model = resnet18(weights='IMAGENET1K_V1')
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载CIFAR-10数据集
cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# 使用数据加载器加载数据
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# 从数据加载器中获取一张图像和标签
original_image, original_label = next(iter(data_loader))

# 定义攻击参数和校正因子
epsilon = 0.03
alpha = 0.0015
iterations = 80

correction_factor = 0.5

# 攻击图像
ground_truth_labels = torch.tensor([2])
perturbed_image = adversarial_attack(model, original_image, label=ground_truth_labels, epsilon=epsilon, alpha=alpha, iterations=iterations)

# 分离攻击和未受攻击的部分
attacked_part, unattacked_part = separate_attack(model, original_image, perturbed_image)

# 校正攻击后的部分

corrected_part = correct_attack(model, original_image, attacked_part, max_iterations=100, lr=0.01)

# 将未受攻击的部分和校正后的部分结合
# not_attacked_part = 1 - attacked_part
combined_image = combine_parts(original_image, unattacked_part, corrected_part)

test(combined_image, original_image)
# 显示结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 6, 1)
plt.imshow(original_image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Original Image")

plt.subplot(1, 6, 2)
plt.imshow(perturbed_image.squeeze().detach().permute(1, 2, 0).detach().numpy())
plt.title("Adversarial Image")

plt.subplot(1, 6, 3)
plt.imshow(torch.clamp(attacked_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("Attacked Part")

plt.subplot(1, 6, 4)
plt.imshow(torch.clamp(unattacked_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("unattacked Part")


plt.subplot(1, 6, 5)
plt.imshow(torch.clamp(corrected_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("Correct Part")

plt.subplot(1, 6, 6)
plt.imshow(combined_image.squeeze().detach().permute(1, 2, 0).detach().numpy())
plt.title("Combined Image")


plt.show()
