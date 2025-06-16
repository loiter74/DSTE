import torch
import torch.nn as nn

class DMF(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, time_steps):
        """
        初始化 DMF 模型
        :param input_channels: 输入的特征维度 (c)
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出特征维度
        :param time_steps: 时间步长 (t)
        """
        super(DMF, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_steps = time_steps

        # 时间特征降维模块
        self.time_encoder = nn.Sequential(
            nn.Linear(time_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 通道特征降维模块
        self.channel_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (b, n, c, t)
        :return: 降维后的特征，形状为 (b, n, output_dim)
        """
        b, n, c, t = x.shape

        # 处理时间维度 (t)
        x_time = x.view(-1, t)  # 将时间维度拉平为 (b*n*c, t)
        x_time = self.time_encoder(x_time)  # 经过时间编码器
        x_time = x_time.view(b, n, c, -1)  # 恢复为 (b, n, c, output_dim)

        # 处理通道维度 (c)
        x_channel = x_time.permute(0, 1, 3, 2).contiguous()  # 调整维度为 (b, n, output_dim, c)
        x_channel = x_channel.view(-1, c)  # 将通道维度拉平为 (b*n*output_dim, c)
        x_channel = self.channel_encoder(x_channel)  # 经过通道编码器
        x_channel = x_channel.view(b, n, -1)  # 恢复为 (b, n, output_dim)

        return x_channel
