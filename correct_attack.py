import torch
import torch.nn as nn
import torch.optim as optim

def correct_attack(model, original_image, perturbed_image, max_iterations=100, lr=0.01):
    # 确保模型处于评估模式
    model.eval()

    # 克隆受攻击的图像并允许梯度计算
    perturbed_image = perturbed_image.clone().detach()
    perturbed_image.requires_grad = True

    # 选择优化器
    optimizer = optim.Adam([perturbed_image], lr=lr)

    for iteration in range(max_iterations):
        # 重置梯度
        optimizer.zero_grad()

        # 获取模型的输出
        perturbed_output = model(perturbed_image)
        original_output = model(original_image)

        # 定义损失函数
        loss = nn.MSELoss()(perturbed_output, original_output)

        # 反向传播
        loss.backward()

        # 优化步骤
        optimizer.step()

        # 可以添加一些停止条件，例如损失下降到一定阈值

    return perturbed_image


