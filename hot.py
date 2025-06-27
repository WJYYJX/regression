import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from trans_res import *


# 定义模型（例如预训练的ResNet18）

backbone = models.resnet18(pretrained=False)
net = ResNet18(backbone, num_classes=1)
model = ResNet18(backbone, num_classes=1)
model.load_state_dict(torch.load('/home/wjy/regression/pytorch_classification/Test5_resnet/xinjiang_1.2.4.6.10.16_nofrezze_res_trans_pretrain_junyunmoniyan.pth_92.7'))
model.eval()  # 切换到评估模式

# 载入图像并进行预处理
img = np.load('/home/wjy/numpydata/xinjiang_select_1.2.4.6.10.16_copy/train/image/-2.5/33baa991f15a4757a9a531996fb97641.npy')
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    transforms.Resize((128, 128),antialias=True)
])
img_tensor = transform(img).unsqueeze(0)  # 扩展维度使其符合模型输入要求

# 注册钩子函数获取卷积层输出
def hook_fn(module, input, output):
    global features
    features = output  # 保存卷积层的输出

# 选择ResNet18中的第一个卷积层
layer = model.conv1
hook = layer.register_forward_hook(hook_fn)

# 进行推理，获得卷积层输出
with torch.no_grad():
    output = model(img_tensor)

# 取消钩子
hook.remove()

# 获取特征图并进行可视化
# features的形状是 (batch_size, channels, height, width)
# 选择某个通道的特征图（例如选择第0个通道）
feature_map = features[0, 5].cpu().numpy()

# 画出热力图
plt.imshow(feature_map, cmap='jet')
plt.colorbar()
plt.savefig("plot5.png")
