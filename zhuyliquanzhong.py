import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
from trans_res import *
import os

# 定义设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 定义模型
backbone = models.resnet18(pretrained=False)
model = ResNet18(backbone, num_classes=1)
model.load_state_dict(torch.load('/home/wjy/regression1/pytorch_classification/Test5_resnet/2restrans_junyunmoniyan_new2_pretrain.pth')) # 替换为您的模型路径
model.eval().to(device)

img_data = np.load('/home/wjy/JUNYUNmoniyandataset_new1/train/image/3.50/POS X -28 Y -30 Z -24.npy')

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

# --- 注意力权重图相关代码 ---
attention_weights = None

# 1. 找到 trans_conv 模块中的 Attention2 实例
attention_layer = None
for name, module in model.named_modules():
    if isinstance(module, Attention2):
        attention_layer = module
        attention_layer_name = name
        break

if attention_layer is None:
    print("Attention2 layer not found!")
    exit()
else:
    print(f"Found Attention2 layer: {attention_layer_name}")

# 2. 定义前向钩子函数来获取注意力权重
def attention_hook(module, input, output):
    global attention_weights
    qkv = module.to_qkv(input[0]).chunk(2, dim=-1)
    q, v = map(lambda t: rearrange(t, 'b n1 n2 (h d) -> b  n1 h n2  d', h=module.heads), qkv)
    k = module.kc(input[0])
    k = rearrange(k, 'b n1 n2 (h d) -> b  n1 h n2  d', h=module.heads)
    dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
    attention_weights = module.attend(dots).detach().cpu()  # 获取注意力权重并移到 CPU


# 3. 注册前向钩子到 Attention2 模块
attention_hook_handle = attention_layer.register_forward_hook(attention_hook)

# 进行前向传播 (不需要梯度计算)
with torch.no_grad():
    _ = model(input_tensor)

# 移除钩子
attention_hook_handle.remove()

# 4. 处理和可视化注意力权重
if attention_weights is not None:
    batch_size, num_channels, num_heads, seq_len_q, seq_len_k = attention_weights.shape

    # seq_len_q 和 seq_len_k 对应于 trans_conv 中 reshape 后的 (h w) 维度
    # 我们需要将其 reshape 回原始图像的空间维度

    # 获取原始图像的尺寸 (假设经过 resize 后是 128x128，原始图像是 H x W x 6)
    img_height, img_width = img_data.shape[:2]
    img_size = 32  # trans_conv 中 size=(32, 32)

    # 创建保存权重图的文件夹
    output_dir = "attention_weight_maps"
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个包含 6 个通道和 num_heads 个头的子图的画布
    fig, axes = plt.subplots(num_channels, num_heads, figsize=(8 * num_heads, 8 * num_channels))

    for channel_idx in range(num_channels):
        original_channel = img_data[:, :, channel_idx]
        normalized_channel = (original_channel - np.min(original_channel)) / (
            np.max(original_channel) - np.min(original_channel) + 1e-8
        )

        for head_idx in range(num_heads):
            attn_map = (
                attention_weights[0, channel_idx, head_idx].mean(dim=0).reshape(img_size, img_size).numpy()
            )
            attn_map_resized = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (img_width, img_height)
            )

            # 保存单独的权重图
            output_path = os.path.join(output_dir, f"channel_{channel_idx + 1}_head_{head_idx + 1}_weights.png")
            plt.imsave(output_path, attn_map_resized_np, cmap='viridis')

            ax = axes[channel_idx, head_idx]
            ax.imshow(normalized_channel, cmap='gray')
            ax.imshow(attn_map_resized, cmap='viridis', alpha=0.5)
            ax.set_title(f'Channel {channel_idx + 1}, Head {head_idx + 1}')
            ax.axis('off')



    plt.tight_layout()
    plt.savefig("attention_weights_per_channel_headmoni3.52_2new.png")
else:
    print("未能获取到 Attention2 的注意力权重！")