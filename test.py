import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from Spatial_Layer import GAT

# 定义超参数
batch_size = 16
learning_rate = 1e-4
num_epochs = 10

num_samples = 100
data_shape = (10, 24, 16)
dummy_data = torch.randn(num_samples, *data_shape)
adj = torch.tensor(np.random.rand(10, 10))

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(dummy_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = GAT(n_feat=16, n_hid=32, n_class=16, dropout=0.2, alpha=0.01, n_heads=4)

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
torch.save(model.state_dict(), 'ds_msa_model.pth')
