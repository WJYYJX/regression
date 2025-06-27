import torch.nn as nn
import torch
from model import resnet101

# regression head


class head(nn.Module):
    def __init__(self,submodel1):
        super(head, self).__init__()

        self.submodel1 = submodel1
        self.conv1 = nn.Conv2d(4096, 4096, kernel_size=3, stride=1, padding=1,
                               bias=False)  # 四层
        #self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.Tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

        # 一个密集层
        self.desens1 = nn.Linear(4096, 1024)
        # 一个剔除层
        self.dropout_layer = nn.Dropout(p=0.5)
        # 两个密集层
        self.desens2 = nn.Linear(1024, 128)
        self.desens3 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.submodel1(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.gelu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.desens1(x)
        #x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.desens2(x)
        #x = self.Tanh(x)
        x = self.desens3(x)
        return x

class regressionhead(nn.Module):
    expansion = 1

    def __init__(self):
        super(regressionhead, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)  # 四层
        #self.bn1 = nn.BatchNorm2d(self.in_channel)
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




def regressionhead():
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    net = resnet101(include_top=False)
    regression = regressionhead()

    combined_model = CombinedModel(net, regression)
    return combined_model()