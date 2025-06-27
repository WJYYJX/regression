import torch.nn as nn
import torch
from model import resnet101
import regressionhead as regressionhead
class SubModel1(nn.Module):
    def __init__(self):
        super(SubModel1, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SubModel2(nn.Module):
    def __init__(self):
        super(SubModel2, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)  # 四层
        self.bn1 = nn.BatchNorm2d(512)
        self.Tanh = nn.Tanh()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

        # 一个密集层
        self.desen1 = nn.Linear(512, 90)
        # 一个剔除层
        self.dropout_layer = nn.Dropout(p=0.5)
        # 两个密集层
        self.desen2 = nn.Linear(90, 30)
        self.desen3 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Tanh(x)
        x = self.conv1(x)
        x = self.Tanh(x)
        x = self.conv1(x)
        x = self.Tanh(x)
        x = self.conv1(x)
        x = self.Tanh(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.desen1(x)
        x = self.dropout_layer(x)
        x = self.desen2(x)
        x = self.desen3(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        self.submodel1 = submodel1
        self.submodel2 = submodel2

    def forward(self, x):
        x = self.submodel1(x)
        x = self.submodel2(x)
        return x


submodel1 = resnet101(include_top=False)
submodel2 = SubModel2()
combined_model = CombinedModel(submodel1, submodel2)
#print(combined_model)
combined_model1 = regressionhead.head(submodel1)
print(combined_model1)