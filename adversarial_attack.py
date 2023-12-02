import torch
import torch.nn as nn

def i_fgsm_attack(image, epsilon, data_grad):
    # 生成扰动并直接添加到图像上，同时限制像素值在[0, 1]范围内
    perturbed_image = torch.clamp(image + epsilon * data_grad.sign(), 0, 1)
    return perturbed_image

def adversarial_attack(model, image, label, epsilon, alpha, iterations):
    """
    I-FGSM迭代快速梯度符号攻击

    参数:
    model: 被攻击的模型。
    image: 输入图像。
    label: 图像对应的真实标签。
    epsilon: 总扰动量。
    alpha: 每次迭代的扰动步长。
    iterations: 总迭代次数。
    """
    # 复制原始图像以创建一个叶子节点
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True

    for i in range(iterations):
        # 正向传播
        output = model(perturbed_image)

        # 计算损失
        loss = torch.nn.CrossEntropyLoss()(output, label)

        # 反向传播，计算梯度
        model.zero_grad()
        loss.backward()

        # 获取图像的梯度
        data_grad = perturbed_image.grad.data

        # 清除当前梯度
        perturbed_image.grad = None

        # 调用FGSM攻击来计算新的扰动图像
        perturbed_image = i_fgsm_attack(perturbed_image, alpha, data_grad)

        # 限制扰动在epsilon范围内
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)

        # 再次设置requires_grad
        perturbed_image = perturbed_image.detach().clone()
        perturbed_image.requires_grad = True

    # 将最终扰动图像的像素值限制在[0, 1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_imag