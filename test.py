import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Spatial_Layer import DS_MSA

# 定义超参数
hidden_dim = 512
num_layers = 6
num_heads = 8
mlp_dim = 1024
dropout_rate = 0.1
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# 创建虚拟数据
# 生成具有您指定维度的随机数据
# 请根据您的具体任务和数据格式进行修改
num_samples = 1000
data_shape = (128, 100, 24, 32)
dummy_data = torch.randn(num_samples, *data_shape)

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(dummy_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = DS_MSA(
    dim=hidden_dim,
    depth=num_layers,
    heads=num_heads,
    mlp_dim=mlp_dim,
    assignment=None,  # 替换为您的分配矩阵
    mask=None,  # 替换为您的掩码
    dropout=dropout_rate
)

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
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # 使用均方误差计算损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'ds_msa_model.pth')
