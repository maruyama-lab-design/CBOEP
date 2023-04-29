import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def NPV_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def read_result(filename):
    df = pd.read_table(filename)
    y_true = df["true"].to_list()
    y_prob = df["pred"].to_list()
    y_pred = list(map(round, y_prob))
    return y_true, y_pred, y_prob




def compare_org_vs_inf(cell="GM12878", train_pos="BG", test="TF"): # 4/14

    plt.figure()

    org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{train_pos}_org_{cell}","prediction", f"masked_{test}_org_{cell}.txt")
    y_true, y_pred, y_prob = read_result(org_file)

    AUC = roc_auc_score(y_true, y_prob)
    MCC = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    height = [AUC, MCC, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=-0.3, label="original")





    inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{train_pos}_INF_{cell}", "prediction", f"masked_{test}_org_{cell}.txt")
    y_true, y_pred, y_prob = read_result(inf_file)

    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    height = [auc, mcc, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")


    plt.xticks(x, ['AUC', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
    plt.ylim((-0.1, 1.2))
    plt.legend()
    plt.title(cell)
    plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"{cell},{train_pos}-{test}.jpg"))



def compare_org_vs_inf_cmn_test(train_cell="GM12878", test_cell="GM12878", dataname="BG"): # 4/14

    plt.figure()

    org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_org_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
    y_true, y_pred, y_prob = read_result(org_file)

    AUC = roc_auc_score(y_true, y_prob)
    MCC = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    height = [AUC, MCC, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=-0.3, label=dataname)





    inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
    y_true, y_pred, y_prob = read_result(inf_file)

    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    height = [auc, mcc, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")

    plt.grid(which = "major", axis = "y", color = "gray", alpha = 0.8, linestyle = "--", linewidth = 1)
    plt.xticks(x, ['AUC', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
    plt.ylim((-0.1, 1.2))
    plt.legend()
    plt.title(f"{train_cell} -> {test_cell} (use common test)")
    plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"org_VS_inf_{train_cell}-{test_cell}_cmn.jpg"))



def compare_2500000_vs_inf_cmn_test(train_cell="GM12878", test_cell="GM12878", dataname="BG"): # 4/14

    plt.figure()

    org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_2500000_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
    y_true, y_pred, y_prob = read_result(org_file)

    AUC = roc_auc_score(y_true, y_prob)
    MCC = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    height = [AUC, MCC, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=-0.3, label="NIMF(d_max=2.5M)")





    inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
    y_true, y_pred, y_prob = read_result(inf_file)

    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    npv = NPV_score(y_true, y_pred)
    spc = specificity_score(y_true, y_pred)

    height = [auc, mcc, b_a, f1, pre, rec, npv, spc] # y_org
    print(height)
    plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")

    plt.grid(which = "major", axis = "y", color = "gray", alpha = 0.8, linestyle = "--", linewidth = 1)
    plt.xticks(x, ['AUC', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
    plt.ylim((-0.1, 1.2))
    plt.legend()
    plt.title(f"{train_cell} -> {test_cell} (use common test)")
    plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"2500000_VS_inf_{train_cell}-{test_cell}_cmn.jpg"))




# compare_org_vs_inf(cell="GM12878")
# compare_org_vs_inf(cell="HeLa-S3")

for cell in ["GM12878", "HeLa-S3", "K562", "NHEK", "IMR90"]:
    compare_2500000_vs_inf_cmn_test(train_cell="GM12878", test_cell=cell, dataname="BG")
    compare_org_vs_inf_cmn_test(train_cell="GM12878", test_cell=cell, dataname="BG")