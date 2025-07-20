
## 环境依赖 

- Python 3.7+
- PyTorch >= 1.8
- numpy, pandas, scikit-learn, tqdm, pyyaml

安装依赖：
```bash
pip install torch numpy pandas scikit-learn tqdm pyyaml
```
---

## 数据准备

将数据放置于 `data/` 目录下的对应子文件夹（如 `data/bjair2017/`）

---



## 模型训练与验证
本流程介绍如何完成神经过程模型的预训练、扩散模型的训练以及测试与验证。相关权重文件会分别保存在 `save/np/` 和 `save/ddpm/` 目录中。

## **1. 神经过程模型的预训练**

运行 `exe_bjair_np.py` 完成神经过程模型的预训练。

### **命令**
```bash
  python exe_bjair_np.py --config base.yaml --pred_attr PM25 --batch_size 32 --epochs 20 --device cuda:0
```
### **参数说明**
- --config：配置文件路径，默认为 base.yaml。
- --pred_attr：预测的空气质量属性，例如 PM25。
- --batch_size：训练时的批量大小，默认为 32。
- --epochs：训练的轮数，默认为 20。
- --device：训练设备，例如 cuda:0 或 cpu。
### **输出**

训练完成后，权重文件会保存在 `save/np/` 目录中，文件名格式为：`model_YYYYMMDDHHMM.pth`。

请记录训练好的模型文件路径（--np_model_path），后续扩散模型训练需要使用该路径。
## **2. 扩散模型的训练**
运行 `exe_bjair_diffusion_with_np.py`，完成扩散模型的训练。

### **命令**
```bash
python exe_bjair_diffusion_with_np.py --config base.yaml --pred_attr PM25 --np_model_path [刚才训练好的np_model路径] --batch_size 64 --epochs 30 --device cuda:0
```
### **参数说明**
- --np_model_path：预训练的神经过程模型路径，请填写第1步保存的权重文件路径。

### **输出**
训练完成后，权重文件会保存在 `save/ddpm/` 目录中，文件名格式为：`model_YYYYMMDDHHMM.pth`。
请记录训练好的扩散模型文件路径（--model_path），后续测试与验证需要使用该路径。
## **3. 测试与验证**
运行 `exe_bjair_diffusion_with_np.py`，加载训练好的扩散模型权重进行测试与验证。

### **命令**
```bash
python exe_bjair_diffusion_with_np.py --config base.yaml --pred_attr PM25 --np_model_path [刚才训练好的np_model路径] --model_path [训练好的扩散模型路径] --test_phase all --device cuda:0
```
### **参数说明**参数说明
- --test_phase：测试阶段，支持以下选项：
    - in：测试分布内数据。
    - out：测试分布外数据。
    - LD_out：测试 LD 分布外数据。
    - all：同时测试所有类型数据。

## 目录结构 
```
DSTE/
│
├─ config/           # 配置文件
├─ data/             # 数据集
├─ model/            # 模型定义
├─ save/             # 训练结果保存
├─ exe_bjair_diffusion.py  # 扩散模型脚本
├─ exe_bjair_np.py         # NP模型脚本
├─ utils.py          # 工具函数
└─ README.md         # 项目说明
```
