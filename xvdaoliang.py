import torch
import torch.nn as nn
import torch.nn.init as init


class MultiChannelLSTM(nn.Module):
    """处理6通道输入的端到端模型

    特征：
    - 输入形状：[batch, 6, H, W] 的6通道数据
    - 每个通道独立特征提取
    - 自适应任意输入尺寸
    - 自动权重初始化
    """

    def __init__(self,
                 in_channels=6,
                 base_channels=16,
                 feature_dim=256,
                 lstm_hidden_size=512,
                 num_lstm_layers=2,
                 dropout=0.3):
        """
        参数：
            in_channels: 输入通道数 (固定为6)
            base_channels: 基础卷积通道数
            feature_dim: 特征输出维度
            lstm_hidden_size: LSTM隐藏层维度
            num_lstm_layers: LSTM堆叠层数
            dropout: LSTM层的dropout概率
        """
        super(MultiChannelLSTM, self).__init__()

        # 参数校验
        assert in_channels == 6, "输入通道数必须为6"

        # 特征提取器
        self.feature_extractors = nn.ModuleList([
            self._build_single_extractor(
                in_channels=1,  # 每个通道单独处理
                base_channels=base_channels,
                output_dim=feature_dim
            ) for _ in range(in_channels)
        ])

        # LSTM模块
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # 初始化权重
        self._init_weights()

    def _build_single_extractor(self, in_channels, base_channels, output_dim):
        """构建单个通道的特征提取器"""
        return nn.Sequential(
            # 空间压缩
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 特征增强
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.MaxPool2d(3, stride=2, padding=1),

            # 特征输出
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, output_dim),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量，形状 [batch, 6, H, W]

        返回：
            output: 预测值 [batch, 1]
        """
        batch_size = x.size(0)

        # 分割通道 [batch, 6, H, W] -> 6个[batch, 1, H, W]
        channels = torch.chunk(x, chunks=6, dim=1)  # 沿通道维度分割

        # 并行特征提取
        features = []
        for i, (extractor, channel_data) in enumerate(zip(self.feature_extractors, channels)):
            # 移除通道维度中的1维度 [batch, 1, H, W] -> [batch, H, W]
            squeezed = channel_data.squeeze(1)  # 现在形状 [batch, H, W]

            # 添加通道维度 [batch, H, W] -> [batch, 1, H, W]
            processed = extractor(squeezed.unsqueeze(1))  # [batch, feature_dim]
            features.append(processed.unsqueeze(1))  # [batch, 1, feature_dim]

        # 堆叠特征序列
        feature_seq = torch.cat(features, dim=1)  # [batch, 6, feature_dim]

        # LSTM处理
        lstm_out, _ = self.lstm(feature_seq)  # lstm_out: [batch, 6, lstm_hidden_size]
        last_state = lstm_out[:, -1, :]  # 取最后时间步 [batch, lstm_hidden_size]

        # 回归预测
        return self.regressor(last_state)

    @torch.no_grad()
    def get_channel_features(self, x):
        """获取各通道的中间特征（用于可视化分析）"""
        channels = torch.chunk(x, chunks=6, dim=1)
        channel_features = []
        for ext, data in zip(self.feature_extractors, channels):
            squeezed = data.squeeze(1).unsqueeze(1)
            feat = ext(squeezed).cpu().numpy()
            channel_features.append(feat)
        return np.stack(channel_features, axis=1)  # [batch, 6, feature_dim]