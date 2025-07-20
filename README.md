# DSTE 项目说明 / DSTE Project README

## 项目简介 / Project Introduction

本项目实现了基于扩散模型（Diffusion Model）和神经过程（Neural Process, NP）的时空数据插补与预测方法，适用于如北京空气质量（BJAir2017）等多种时空数据集。

---

## 环境依赖 / Requirements

- Python 3.7+
- PyTorch >= 1.8
- numpy, pandas, scikit-learn, tqdm, pyyaml

安装依赖：
pip install torch numpy pandas scikit-learn tqdm pyyaml

---

## 数据准备 / Data Preparation

将数据放置于 `data/` 目录下的对应子文件夹（如 `data/bjair2017/`），确保数据格式符合代码要求。

---

## 主要脚本与运行说明 / Main Scripts & Usage

### 1. 扩散模型训练与测试 / Diffusion Model Training & Testing

运行扩散模型主脚本：
python exe_bjair_diffusion.py --config base.yaml --pred_attr PM25 --batch_size 64 --epochs 30 --device cuda:0

**运行说明：**
- `exe_bjair_diffusion.py` 是扩散模型的训练与测试脚本。
- 参数 `--config` 指定配置文件；`--pred_attr` 指定预测目标（如 PM2.5）；`--batch_size` 和 `--epochs` 分别控制训练批量大小和轮数；`--device` 指定运行设备（如 GPU 或 CPU）。
- 若提供 `--model_path` 参数，则加载预训练模型进行评估，否则从头开始训练。

---

### 2. 神经过程模型训练与测试 / Neural Process Model Training & Testing

运行 NP 模型主脚本：
python exe_bjair_np.py --config base.yaml --pred_attr PM25 --batch_size 32 --epochs 20 --device cuda:0

**运行说明：**
- `exe_bjair_np.py` 是神经过程模型的训练与测试脚本。
- 参数设置与扩散模型类似。通常先运行此脚本完成 NP 模型的预训练，生成权重文件供扩散模型使用。

---

## 训练与测试流程 / Workflow

1. 配置好 `config/base.yaml`，并准备好数据。
2. 运行 `exe_bjair_np.py` 完成神经过程模型的预训练，权重文件会保存在 `save/np/` 目录。
3. 运行 `exe_bjair_diffusion.py` 进行扩散模型的训练与测试，权重文件会保存在 `save/ddpm/` 目录。
4. 训练和测试结果（如损失、CRPS 等指标）会在终端输出。

---

## 目录结构 / Directory Structure

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

---

## 结果输出 / Output

- 训练过程中的指标（如损失、CRPS）会在终端实时输出。
- 模型权重文件分别保存在 `save/np/` 和 `save/ddpm/` 目录。
- 可视化结果可通过修改 `utils.py` 中的 `plot_groups()` 函数实现。

---

## 联系与贡献 / Contact & Contribution

如有问题或建议，欢迎提交 issue 或 pull request。
