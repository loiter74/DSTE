## Environment Dependencies

- Python 3.7+
- PyTorch >= 1.8
- numpy, pandas, scikit-learn, tqdm, pyyaml

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn tqdm pyyaml
```

## Data Preparation

Place the data in the corresponding subfolder under the `data/` directory (e.g., `data/bjair2017/`).

## Model Training and Validation

This process introduces how to complete the pre-training of the neural process model, the training of the diffusion model, and testing and validation. Relevant weight files will be saved in the `save/np/` and `save/ddpm/` directories respectively. We have prepared some data, and you can follow the steps below to train and test.

### **1. Pre-training of Neural Process Model**

Run `exe_bjair_np.py` to complete the pre-training of the neural process model.

#### **Command**
```bash
  python exe_bjair_np.py --pred_attr PM25 --epochs 20
```
#### **Parameter Description**
- `--config`: Path to the configuration file, default is base.yaml
- `--pred_attr`: Predicted air quality attribute, e.g., PM25
- `--batch_size`: Batch size during training, default is 32
- `--epochs`: Number of training epochs, default is 20
- `--device`: Training device, e.g., cuda:0 or cpu
#### **Output**

After training, the weight file will be saved in the `save/np/` directory. The filename format is: `model_YYYYMMDDHHMM.pth`.

Please record the path of the trained model file (`--np_model_path`), as it will be needed for the subsequent diffusion model training.

### **2. Diffusion Model Training**

Run `exe_bjair_diffusion_with_np.py` to complete the training of the diffusion model.

#### **Command**
```bash
python exe_bjair_diffusion_with_np.py --pred_attr PM25 --np_model_path [Path to the previously trained np_model] --epochs 30
```
#### **Parameter Description**
- `--np_model_path`: Path to the pre-trained neural process model, please provide the weight file path saved in step 1.

#### **Output**
After training, the weight file will be saved in the `save/ddpm/` directory. The filename format is: `model_YYYYMMDDHHMM.pth`.

Please record the path of the trained diffusion model file (`--model_path`), as it will be needed for subsequent testing and validation.

### **3. Testing and Validation**

Run `exe_bjair_diffusion_with_np.py`, load the trained diffusion model weights for testing and validation.

#### **Command**
```bash
python exe_bjair_diffusion_with_np.py --config base.yaml --pred_attr PM25 --np_model_path [Path to the previously trained np_model] --model_path [Path to the trained diffusion model] --test_phase all --device cuda:0
```
#### **Parameter Description**
- `--test_phase`: Testing phase, supports the following options:
    - in: Test in-distribution data
    - out: Test out-of-distribution data
    - LD_out: Test LD out-of-distribution data
    - all: Test all types of data simultaneously

## Directory Structure
```
DSTE/
│
├─ config/           
├─ data/             # Dataset
├─ model/            
├─ save/             # Training results saved here
├─ exe_bjair_diffusion.py  
├─ exe_bjair_np.py         
├─ utils.py          
└─ README.md         
