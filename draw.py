# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:43:28 2024

@author: LUN076
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def draw_ddpm(n, y, label, mask, time, data_dir=None):
    val = 1-mask
    plt.plot([i for i in range(len(label))], label, 'r-', linewidth=0.5, label='True Label')
    plt.plot([i for i in range(len(label))], y.median(dim=0).values*val+label*(1-val), 'o-', linewidth=0.5, label='Model Median')
    # 绘制上下界
    range_multiplier = 2
    variance = torch.var(y, dim=0)
    mean_value = torch.mean(y, dim=0)
    lower_bound = mean_value - range_multiplier * torch.sqrt(variance)
    upper_bound = mean_value + range_multiplier * torch.sqrt(variance)
    plt.fill_between([i+begin for i in range(len(label))],
                     lower_bound,
                     upper_bound,
                     color='orange', alpha=0.5, label='Prediction Interval')
    # 设置图像标题和标签
    plt.title('Model Prediction with Bounds')
    plt.xlabel('Index n:{}'.format( n))
    plt.ylabel('Value')

    # 显示图例，并设置图例大小
    legend = plt.legend(loc='upper right', framealpha=0.3, fontsize=5)
    # 调整图例的大小
    legend.get_frame().set_linewidth(0.4)  # 设置图例边框宽度
    legend.get_frame().set_edgecolor('gray')  # 设置图例边框颜色
    legend.get_frame().set_facecolor('white')  # 设置图例背景色
    plt.margins(x=0)
    # 显示网格线
    plt.grid(True)

    if data_dir is not None:
        if not os.path.exists(data_dir + 'pictures/'):
            os.makedirs(data_dir + 'pictures/')
        plt.savefig(data_dir + 'pictures/b{}_n{}.png'.format(b, n), dpi=300)
    plt.show()


def draw(y, label, mask, save_dir=None):
    y = y[0,0,0,:].cpu().detach().numpy()
    label = label[0,0,0,:].cpu().detach().numpy()
    mask = mask[0,0,0,:].cpu().detach().numpy()
    val = 1-mask
    plt.plot( [i for i in range(len(label))], label, 'r-', linewidth=0.1, label='True Label')
    plt.plot( [i for i in range(len(label))], y*val+label*(1-val), 'o-', linewidth=0.1, label='Pred', markersize=0.2)

    plt.margins(x=0)
    # 显示网格线
    plt.grid(True)

    if save_dir is not None:
        if not os.path.exists(save_dir + 'pictures/'):
            os.makedirs(save_dir + 'pictures/')
        plt.savefig(save_dir + 'pictures/b{}_n{}.png'.format(b, n), dpi=300)
    plt.show()


def plot_line(x, y, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    绘制简洁的折线图

    参数:
        x: x轴数据
        y: y轴数据
        title: 图表标题（可选）
        xlabel: x轴标签（可选）
        ylabel: y轴标签（可选）
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_pm25_heatmap(time_data, pred_data, feature_idx=0, title="PM2.5浓度热力图", xlabel="时间", ylabel="站点",
                      save_path=None, cmap="viridis", figsize=(15, 8)):
    """
    绘制时间-站点的PM2.5浓度热力图，支持中文显示

    参数:
    time_data: 时间数据数组
    pred_data: 形状为 [num_stations, num_timesteps, num_features] 的预测数据
    feature_idx: 要绘制的特征索引，默认为0（第一个特征）
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    save_path: 保存图像的路径，如果为None则不保存
    cmap: 热力图使用的颜色映射
    figsize: 图像大小
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import platform

    # 设置中文字体
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑和黑体
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 提取要绘制的特征数据
    # 如果数据是 [num_stations, num_timesteps, num_features]，选择指定的特征
    if len(pred_data.shape) == 3:
        data_to_plot = pred_data[:, :, feature_idx]
    else:
        data_to_plot = pred_data  # 如果已经是二维数据，直接使用

    num_stations, num_timesteps = data_to_plot.shape


    # 创建图形
    plt.figure(figsize=figsize)

    # 绘制热力图 - 注意这里使用的是 data_to_plot
    im = plt.imshow(data_to_plot, aspect='auto', cmap=cmap, interpolation='none')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('PM2.5浓度 (归一化值)')

    # 设置坐标轴
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 设置标题
    plt.title(title)

    # 处理时间轴标签

    if time_data is not None and len(time_data) > 0:
        # 选择适当的时间间隔来显示标签
        num_ticks = min(10, num_timesteps)  # 最多显示10个时间标签
        tick_indices = np.linspace(0, num_timesteps - 1, num_ticks, dtype=int)

        plt.xticks(tick_indices, [time_data[i] for i in tick_indices], rotation=45)

    # 设置站点标签
    station_ticks = np.linspace(0, num_stations - 1, min(10, num_stations), dtype=int)
    plt.yticks(station_ticks, [f"站点 {i + 1}" for i in station_ticks])

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == '__main__':
    import matplotlib.dates as mdates
    data_dir = "save/BJAir/202501200616PM25_Concentration/"

    label = torch.load(data_dir + "label_tensor.pt").squeeze(2)
    time_utc = torch.load(data_dir + "time_utc_tensor.pt")
    y = torch.load(data_dir + "y_tensor.pt").squeeze(2)
    val_index = torch.load(data_dir + '/val_index_tensor.pt').squeeze(2)
    b, n, t = label.shape

    true_label = []
    predicted = []
    time = []
    val = []

    begin = 0
    for b_i in range(b):
        for n_i in range(n):

            draw(n_i, y[b_i][n_i], label[b_i][n_i], val_index[b_i][n_i],  time_utc[b_i], data_dir, begin)
            true_label += label[b_i][n_i].flatten(-1)
            predicted += y[b_i][n_i].flatten(-1)
            val += val_index[b_i][n_i].flatten(-1)
            time.append([begin+i for i in range(len(label[b_i][n_i]))])
            begin += len(label[b_i][n_i])
    draw(0, torch.tensor(predicted[::24]), torch.tensor(true_label[::24]), torch.tensor(val[::24]), time, data_dir)





        

        
        
        
        
        
        