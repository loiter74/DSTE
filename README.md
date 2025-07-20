
## 环境依赖 

- Python 3.7+
- PyTorch >= 1.8
- numpy, pandas, scikit-learn, tqdm, pyyaml

安装依赖：
```bash
pip install torch numpy pandas scikit-learn tqdm pyyaml

---

## 数据准备 / Data Preparation

将数据放置于 `data/` 目录下的对应子文件夹（如 `data/bjair2017/`），确保数据格式符合代码要求。

## 数据准备

## 主要脚本与运行说明 / Main Scripts & Usage


## 模型训练与验证
本流程介绍如何完成神经过程模型的预训练、扩散模型的训练以及测试与验证。相关权重文件会分别保存在 `save/np/` 和 `save/ddpm/` 目录中。

**运行说明：**
- `exe_bjair_diffusion.py` 是扩散模型的训练与测试脚本。
- 参数 `--config` 指定配置文件；`--pred_attr` 指定预测目标（如 PM2.5）；`--batch_size` 和 `--epochs` 分别控制训练批量大小和轮数；`--device` 指定运行设备（如 GPU 或 CPU）。
- 若提供 `--model_path` 参数，则加载预训练模型进行评估，否则从头开始训练。

运行 `exe_bjair_np.py` 完成神经过程模型的预训练。

### 2. 神经过程模型训练与测试 / Neural Process Model Training & Testing

运行 NP 模型主脚本：
python exe_bjair_np.py --config base.yaml --pred_attr PM25 --batch_size 32 --epochs 20 --device cuda:0

训练完成后，权重文件会保存在 `save/np/` 目录中，文件名格式为：`model_YYYYMMDDHHMM.pth`。

---

## 训练与测试流程 / Workflow

1. 配置好 `config/base.yaml`，并准备好数据。
2. 运行 `exe_bjair_np.py` 完成神经过程模型的预训练，权重文件会保存在 `save/np/` 目录。
3. 运行 `exe_bjair_diffusion.py` 进行扩散模型的训练与测试，权重文件会保存在 `save/ddpm/` 目录。
4. 训练和测试结果（如损失、CRPS 等指标）会在终端输出。

---

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
