from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

dirlist = glob.glob("D:\\ylwrv\\\Koga_code\\TransEPI\\output\\*")
# dirlist = ["BENGI", "maxflow", "BENGI_noMSE", "maxflow_noMSE"]

for dirname in dirlist:
    if not os.path.exists(f"{dirname}//loss.csv"):
        continue
    loss_df = pd.read_csv(f"{dirname}//loss.csv")

    loss_df["epochs"] = loss_df["epochs"].astype(int)
    x_data = loss_df["epochs"].tolist()
    y_data1 = loss_df["train_loss"].tolist()
    y_data2 = loss_df["valid_loss"].tolist()

    fig = plt.figure()

    plt.ylim((0, 1))
    plt.plot(x_data, y_data1, label="train loss")
    plt.plot(x_data, y_data2, label="valid loss")
    plt.legend()
    plt.title(f"{os.path.basename(dirname)}")
    fig.savefig(f"{dirname}/loss.png")