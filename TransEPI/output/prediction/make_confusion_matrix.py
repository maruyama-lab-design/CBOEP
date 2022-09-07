from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd


dirname = "D:\\ylwrv\\Koga_code\\TransEPI\\output\\prediction\\0903"
filenames = glob.glob(os.path.join(dirname, "*prediction.txt"))
for file in filenames:

    basename = os.path.basename(file)
    result_df = pd.read_table(file, header=None, index_col=None, skiprows=2, names=["true", "pred", "chrom", "enh_name", "prom_name"])
    print(result_df.head())
    y_true = result_df["true"].tolist()
    y_pred = result_df["pred"].tolist()

    y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    cm = confusion_matrix(y_true, y_pred)

    print(cm)

    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt="d")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig(os.path.join(dirname, "cm", f"{basename}_cm.png"))