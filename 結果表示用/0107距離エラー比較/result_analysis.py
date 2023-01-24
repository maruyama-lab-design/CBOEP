from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt







# 'd:\\ylwrv\\Koga_code\\結果表示用\\0107距離エラー比較\\result\\use_distance\\BENGI_orginal_GM12878_noScheduler(lr=0.0001)\\GM12878-GM12878_prediction.txt'
# 'D:\\ylwrv\\Koga_code\\結果表示用\\0107距離エラー比較\\result\\use_distance\\BENGI_original_GM12878_noScheduler(lr=0.0001)\GM12878-GM12878_prediction.txt'

def make_table(train_cell):
    use_d_dirname = os.path.join(os.path.dirname(__file__), "result", "use_distance", f"BENGI_original_{train_cell}_noScheduler(lr=0.0001)")
    no_d_dirname = os.path.join(os.path.dirname(__file__), "result", "no_distance", f"BENGI_original_{train_cell}_noScheduler(lr=0.0001)_noMSE")
    table_list = [] # "test cell", "MCC old", "MCC new", "MCC new-old", "bl-acc old", "bl-acc new", "bl-acc new-old"
    for test_cell in ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]:
        use_d_filename = os.path.join(use_d_dirname, f"{train_cell}-{test_cell}_prediction.txt")
        no_d_filename = os.path.join(no_d_dirname, f"{train_cell}-{test_cell}_prediction.txt")

        new_df = pd.read_table(use_d_filename, header=None, index_col=None, skiprows=1, names=["true", "prob"])
        n_true = new_df["true"].to_list()
        n_prob = new_df["prob"].to_list()
        n_pred =  list(map(round, n_prob))
        n_ba = mt.balanced_accuracy_score(n_true, n_pred)
        n_mcc = mt.matthews_corrcoef(n_true, n_pred)

        old_df = pd.read_table(no_d_filename, header=None, index_col=None, skiprows=1, names=["true", "prob"])
        o_true = old_df["true"].to_list()
        o_prob = old_df["prob"].to_list()
        o_pred =  list(map(round, o_prob))
        o_ba = mt.balanced_accuracy_score(o_true, o_pred)
        o_mcc = mt.matthews_corrcoef(o_true, o_pred)

        table_list.append([test_cell, o_mcc, n_mcc, o_ba, n_ba])
    table = pd.DataFrame(table_list, columns=["test cell", "MCC no dist", "MCC use dist", "bl-acc no dist", "bl-acc use dist"])
    table.to_csv(os.path.join(os.path.dirname(__file__), "table", f"train_cell={train_cell}.csv"), index=False)
    with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), "table", f"TransEPI_distance_error.xlsx"), mode="a", if_sheet_exists="replace") as writer:
        table.to_excel(writer, sheet_name=f'train={train_cell}')




def make_bar(train_cl):
    pass



for train_cell in ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]:
    make_table(train_cell)