import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from trans_res import ResNet18

# 定义设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 定义模型
backbone = models.resnet18(pretrained=False)
model = ResNet18(backbone, num_classes=1)
model.load_state_dict(torch.load('/home/wjy/regression1/pytorch_classification/Test5_resnet/2restrans_junyunmoniyan_new2_pretrain.pth'))
model.eval() # 保持评估模式，但允许计算梯度
model.to(device)

# 载入六通道灰度图数据
img_data = np.load('/home/wjy/JUNYUNmoniyandataset——new/train/image/-2.50/POS X -12 Y -10 Z -12.npy')

# 假设 img_data 的形状是 (height, width, 6)
# 预处理：转换为 Tensor 并调整通道顺序
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为 Tensor, 形状为 (C, H, W)
    transforms.Resize((128, 128), antialias=True)
])

# 创建一个包含 6 个灰度通道的 PIL Image 列表
images = []
for i in range(img_data.shape[2]):
    channel = (img_data[:, :, i] * 255).astype(np.uint8)
    image = Image.fromarray(channel, mode='L') # 'L' 表示灰度图像
    images.append(image)

# 将 PIL Image 列表转换为一个多通道 Tensor
img_tensor_list = [transform(img) for img in images] # 每个元素形状为 (1, 128, 128)
img_tensor = torch.stack(img_tensor_list, dim=0).unsqueeze(0) # 堆叠得到 (6, 1, 128, 128)，然后添加批次维度 -> (1, 6, 1, 128, 128)

# 修改这里，直接将 PIL 图像列表堆叠，然后调整通道顺序
img_tensor_list = [transform(img) for img in images] # 每个元素形状为 (1, 128, 128)
img_tensor = torch.cat(img_tensor_list, dim=0).reshape(1, 6, 128, 128) # 沿通道维度拼接，然后 reshape

input_tensor = img_tensor.requires_grad_(True).to(device) # 确保输入需要梯度

# 确定最后一个卷积层 (请根据您的模型结构进行调整)
last_conv_layer = None
last_conv_layer_name = None
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        last_conv_layer = module
        last_conv_layer_name = name

print(f"使用最后一个卷积层: {last_conv_layer_name}")

# 存储最后一个卷积层的特征图和梯度
feature_maps = None
gradients = None

def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def full_backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# 注册钩子
forward_handle = last_conv_layer.register_forward_hook(forward_hook)
backward_handle = last_conv_layer.register_full_backward_hook(full_backward_hook)

# 进行推理 (移除 no_grad 上下文)
output = model(input_tensor)

# 获取目标类别的得分 (对于回归任务，可以直接使用输出的第一个元素)
target_output = output[0]

# 计算目标输出相对于最后一个卷积层特征图的梯度
model.zero_grad()
target_output.backward()

# 移除钩子
forward_handle.remove()
backward_handle.remove()

# 获取特征图和梯度
pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
feature_maps = feature_maps.detach().cpu() # 将特征图移回 CPU 进行后续处理
pooled_gradients = pooled_gradients.detach().cpu() # 将梯度移回 CPU

# 将梯度作为权重应用于特征图
grad_cam_map = torch.sum(pooled_gradients * feature_maps, dim=1, keepdim=True)
grad_cam_map = F.relu(grad_cam_map)
grad_cam_map = grad_cam_map.squeeze().numpy()

# 获取原始图像的尺寸 (假设每个通道的尺寸相同)
img_height, img_width = img_data.shape[:2]

# 为每个原始通道生成并显示 Grad-CAM 热力图
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    original_channel = img_data[:, :, i]
    normalized_channel = (original_channel - np.min(original_channel)) / (np.max(original_channel) - np.min(original_channel) + 1e-8)

    grad_cam_map_resized = Image.fromarray(grad_cam_map).resize((img_width, img_height))
    grad_cam_map_resized = np.array(grad_cam_map_resized)

    axes[i].imshow(normalized_channel, cmap='gray')
    axes[i].imshow(grad_cam_map_resized, cmap='jet', alpha=0.5)
    axes[i].set_title(f'Channel {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("grad_cam_per_channelmoni1.png")
plt.show()