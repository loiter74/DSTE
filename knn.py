import argparse
from multiprocessing import freeze_support

import torch
from sklearn.neighbors import KNeighborsRegressor
from data.dataset_BJAir2017 import get_dataloader
# from data.dataset_PEMS03 import get_dataloader
from utils import _mae_with_missing, _rmse_with_missing, _mape_with_missing, get_info

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description="DSTE")
    parser.add_argument('--pred_attr', type=str, default="PM10_Concentration", help='Which AQ attribute to infer')
    parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_train_target', type=int, default=3, help='# of target set for training at the start')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    # parser.add_argument('--missing_rate', type=float, default=0.6)

    opt = parser.parse_args()

    train_loader, valid_loader, test_loader = get_dataloader(opt)

    knn = KNeighborsRegressor(n_neighbors=3)
    for data in train_loader:  # inner loop within one epoch
        pred_context = data["pred_context"].squeeze(-1)
        pred_target = data["pred_target"].squeeze(-1)
        b, m, t = pred_context.shape
        for i in range(pred_target.shape[1]):
            X = pred_context.reshape(b*t, m)
            y = pred_target[:, i, :].reshape(b*t, -1).squeeze(-1)
            knn.fit(X, y)

    y = []
    y_pred = []
    mask = []
    for data in test_loader:
        pred_context = data["pred_context"].squeeze(-1)
        pred_target = data["pred_target"].squeeze(-1)
        b, m, t = pred_context.shape
        for i in range(pred_target.shape[1]):
            X = pred_context.reshape(b * t, m)
            y.append(pred_target[:, i, :].reshape(b * t, -1).squeeze(-1))
            mask.append(data["missing_mask_target"][:, i, :].reshape(b * t, -1).squeeze(-1))
            y_pred.append(torch.tensor(knn.predict(X)))

    print("scale: ", opt.scale)
    print("mean: ", opt.mean)
    print("mae: ",_mae_with_missing(torch.concat(y_pred) * opt.scale, torch.concat(y) * opt.scale, torch.concat(mask)))
    print("rmse: ",_rmse_with_missing(torch.concat(y_pred) * opt.scale, torch.concat(y) * opt.scale, torch.concat(mask)))
    print("mape: ",_mape_with_missing(torch.concat(y_pred) * opt.scale + opt.mean, torch.concat(y) * opt.scale + opt.mean,torch.concat(mask)))
