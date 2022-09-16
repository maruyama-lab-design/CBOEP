# learning rate を plot するためのコード
import torch
from torch import nn
import matplotlib.pyplot as plt

model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))

# 学習率表示用
def lr_plot(scheduler):
    lrs = []
    for i in range(100):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)