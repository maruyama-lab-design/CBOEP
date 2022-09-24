from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy


dirname = "D:\\ylwrv\\Koga_code\\TransEPI\\output\\prediction\\0916"
filenames = glob.glob(os.path.join(dirname, "*prediction.txt"))
for file in filenames:

    basename = os.path.basename(file)
    result_df = pd.read_table(file, header=None, index_col=None, skiprows=2, names=["true", "pred", "chrom", "enh_name", "prom_name"])
    print(result_df.head())
    y_true = result_df["true"].tolist()
    y_pred = result_df["pred"].tolist()

    all_cnt = len(y_true)

    y_pred = [1.0 if i >= 0.5 else 0.0 for i in y_pred]
    cm = confusion_matrix(y_true, y_pred)

    print(cm)
    cm = cm.astype(numpy.float32)

    cm[0][0]  = cm[0][0] / all_cnt
    cm[0][1]  = cm[0][1] / all_cnt
    cm[1][0]  = cm[1][0] / all_cnt
    cm[1][1]  = cm[1][1] / all_cnt
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".3f")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig(os.path.join(dirname, "cm", f"{basename}_cm.png"))