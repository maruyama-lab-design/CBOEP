from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt










def make_table(train_cell):
    new_dirname = os.path.join(os.path.dirname(__file__), "result", "new", f"BG_org(cl_wise)_{train_cell}_noScheduler(lr=0.0001)_noMSE_mask-wn")
    old_dirname = os.path.join(os.path.dirname(__file__), "result", "old", f"BG_org(cl_wise)_{train_cell}_noScheduler(lr=0.0001)_noMSE_mask-wn_old")
    table_list = [] # "test cell", "MCC old", "MCC new", "MCC new-old", "bl-acc old", "bl-acc new", "bl-acc new-old"
    for test_cell in ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]:
        new_filename = os.path.join(new_dirname, f"{train_cell}-{test_cell}_prediction.txt")
        old_filename = os.path.join(old_dirname, f"{train_cell}-{test_cell}_prediction.txt")

        new_df = pd.read_table(new_filename, header=None, index_col=None, skiprows=1, names=["true", "prob"])
        n_true = new_df["true"].to_list()
        n_prob = new_df["prob"].to_list()
        n_pred =  list(map(round, n_prob))
        n_ba = mt.balanced_accuracy_score(n_true, n_pred)
        n_mcc = mt.matthews_corrcoef(n_true, n_pred)

        old_df = pd.read_table(old_filename, header=None, index_col=None, skiprows=1, names=["true", "prob"])
        o_true = old_df["true"].to_list()
        o_prob = old_df["prob"].to_list()
        o_pred =  list(map(round, o_prob))
        o_ba = mt.balanced_accuracy_score(o_true, o_pred)
        o_mcc = mt.matthews_corrcoef(o_true, o_pred)

        table_list.append([test_cell, o_mcc, n_mcc, n_mcc-o_mcc, o_ba, n_ba, n_ba-o_ba])
    table = pd.DataFrame(table_list, columns=["test cell", "MCC old", "MCC new", "MCC new-old", "bl-acc old", "bl-acc new", "bl-acc new-old"])
    table.to_csv(os.path.join(os.path.dirname(__file__), "table", f"train_cell={train_cell}.csv"), index=False)
    with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), "table", f"TransEPI_making.xlsx"), mode="a") as writer:
        table.to_excel(writer, sheet_name=f'train={train_cell}')




def make_bar(train_cl):
    pass



for train_cell in ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]:
    make_table(train_cell)