import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from net_model import ST_Module

# 定义超参数
batch_size = 16
learning_rate = 1e-4
num_epochs = 100

c, n, t = 4, 10, 24

num_samples = 1000
data_shape = (c, n, t)
dummy_data = torch.ones(num_samples, *data_shape)
adj = torch.tensor(np.random.rand(n, n))

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(dummy_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义模型文件的路径
model_path = 'my_model.pth'

# 检查模型文件是否存在
if os.path.exists(model_path):
    # 如果模型文件存在，则加载已有的模型
    model = torch.load(model_path)
    print("已加载已有的模型。")
else:
    # 创建模型
    model = ST_Module(c, 2*c, t, st_blocks=3) # num_time的意义是时间编码)
    print("已创建新的模型。")

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为示例损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_data in train_loader:
        inputs = batch_data[0]  # 获取输入数据
        optimizer.zero_grad()
        outputs = model(inputs, adj)
        loss = criterion(outputs, inputs)  # 使用均方误差计算损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'st_model.pth')
