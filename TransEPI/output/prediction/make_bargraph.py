# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt





from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt

def pr_auc_score(y_true, y_prob):
    precision, recall, _ = mt.precision_recall_curve(y_true, y_prob)
    aupr = mt.auc(recall, precision)
    return aupr


def specifity_score(y_true, y_pred):
    tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

    # NPV
    # return tn / (tn + fn)

    # specificity
    #  TN / TN + FP
    return tn / (tn + fp)


def NPV_score(y_true, y_pred):
    tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

    # NPV
    return tn / (tn + fn)


def TargetFinder_result_analysis():
    filename1 = "train_TF_org(e=1),test_TF_org.prediction.txt"
    filename2= "train_TF_mf(e=9),test_TF_mf.prediction.txt"

    df = pd.read_table(filename1, header=None, index_col=None, skiprows=2, names=["true", "prob", "chrom", "enh_name", "prm_name"])
    true = df["true"].to_list()
    prob = df["prob"].to_list()

    auc = mt.roc_auc_score(true, prob)
    aupr = pr_auc_score(true, prob)
    pred =  list(map(round, prob))
    true_prime = [int(i == 0) for i in true]
    pred_prime = [int(i == 0) for i in pred]
    f1 = mt.f1_score(true, pred)
    f1_prime = mt.f1_score(true_prime, pred_prime)
    ba = mt.balanced_accuracy_score(true, pred)
    pre = mt.precision_score(true, pred)
    rec = mt.recall_score(true, pred)
    spc = specifity_score(true, pred)
    npv = NPV_score(true, pred)
    mcc = mt.matthews_corrcoef(true, pred)
    
    # auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc
    y_org = [auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc]
    x_org = [i+1 for i in range(len(y_org))]




    df = pd.read_table(filename2, header=None, index_col=None, skiprows=2, names=["true", "prob", "chrom", "enh_name", "prm_name"])
    true = df["true"].to_list()
    prob = df["prob"].to_list()

    auc = mt.roc_auc_score(true, prob)
    aupr = pr_auc_score(true, prob)
    pred =  list(map(round, prob))
    true_prime = [int(i == 0) for i in true]
    pred_prime = [int(i == 0) for i in pred]
    f1 = mt.f1_score(true, pred)
    f1_prime = mt.f1_score(true_prime, pred_prime)
    ba = mt.balanced_accuracy_score(true, pred)
    pre = mt.precision_score(true, pred)
    rec = mt.recall_score(true, pred)
    spc = specifity_score(true, pred)
    npv = NPV_score(true, pred)
    mcc = mt.matthews_corrcoef(true, pred)
    
    # auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc
    y_mf = [auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc]
    x_mf = [i+1.3 for i in range(len(y_org))]


    label_x = ["AUC", "AUPR", "F(epi)", "F(no-epi)", "balanced\naccuracy", "precison", "recall", "specificity", "NPV", "MCC"]

    fig = plt.figure(figsize = (20,5))
    plt.ylim((0, 1))

    # 1つ目の棒グラフ
    plt.bar(x_org, y_org, color='b', width=0.3, label='TF_org', align="center")

    # 2つ目の棒グラフ
    plt.bar(x_mf, y_mf, color='r', width=0.3, label='TF_mf', align="center")

    # 凡例
    plt.legend(loc=2)

    # X軸の目盛りを置換
    plt.xticks([i+1.15 for i in range(len(y_org))], label_x)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(f"figure/TargetFinder_result.png", dpi=300, transparent=True)





def BENGI_result_analysis():
    filename1 = "train_org(e=1),test_org.prediction.txt"
    filename2= "train_mf(e=1),test_mf.prediction.txt"

    df = pd.read_table(filename1, header=None, index_col=None, skiprows=2, names=["true", "prob", "chrom", "enh_name", "prm_name"])
    true = df["true"].to_list()
    prob = df["prob"].to_list()

    auc = mt.roc_auc_score(true, prob)
    aupr = pr_auc_score(true, prob)
    pred =  list(map(round, prob))
    true_prime = [int(i == 0) for i in true]
    pred_prime = [int(i == 0) for i in pred]
    f1 = mt.f1_score(true, pred)
    f1_prime = mt.f1_score(true_prime, pred_prime)
    ba = mt.balanced_accuracy_score(true, pred)
    pre = mt.precision_score(true, pred)
    rec = mt.recall_score(true, pred)
    spc = specifity_score(true, pred)
    npv = NPV_score(true, pred)
    mcc = mt.matthews_corrcoef(true, pred)
    
    # auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc
    y_org = [auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc]
    x_org = [i+1 for i in range(len(y_org))]




    df = pd.read_table(filename2, header=None, index_col=None, skiprows=2, names=["true", "prob", "chrom", "enh_name", "prm_name"])
    true = df["true"].to_list()
    prob = df["prob"].to_list()

    auc = mt.roc_auc_score(true, prob)
    aupr = pr_auc_score(true, prob)
    pred =  list(map(round, prob))
    true_prime = [int(i == 0) for i in true]
    pred_prime = [int(i == 0) for i in pred]
    f1 = mt.f1_score(true, pred)
    f1_prime = mt.f1_score(true_prime, pred_prime)
    ba = mt.balanced_accuracy_score(true, pred)
    pre = mt.precision_score(true, pred)
    rec = mt.recall_score(true, pred)
    spc = specifity_score(true, pred)
    npv = NPV_score(true, pred)
    mcc = mt.matthews_corrcoef(true, pred)
    
    # auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc
    y_mf = [auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc]
    x_mf = [i+1.3 for i in range(len(y_org))]


    label_x = ["AUC", "AUPR", "F(epi)", "F(no-epi)", "balanced\naccuracy", "precison", "recall", "specificity", "NPV", "MCC"]

    fig = plt.figure(figsize = (20,5))
    plt.ylim((0, 1))

    # 1つ目の棒グラフ
    plt.bar(x_org, y_org, color='b', width=0.3, label='BG_org', align="center")

    # 2つ目の棒グラフ
    plt.bar(x_mf, y_mf, color='r', width=0.3, label='BG_mf', align="center")

    # 凡例
    plt.legend(loc=2)

    # X軸の目盛りを置換
    plt.xticks([i+1.15 for i in range(len(y_org))], label_x)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(f"figure/BENGI_result.png", dpi=300,transparent=True)

TargetFinder_result_analysis()
BENGI_result_analysis()




# dataname = "TargetFinder"


# x1 = [1, 2, 3]
# y1 = [4, 5, 6]

# x2 = [1.3, 2.3, 3.3]
# y2 = [2, 4, 1]

# label_x = ['Result1', 'Result2', 'Result3']

# # 1つ目の棒グラフ
# plt.bar(x1, y1, color='b', width=0.3, label='Data1', align="center")

# # 2つ目の棒グラフ
# plt.bar(x2, y2, color='g', width=0.3, label='Data2', align="center")

# # 凡例
# plt.legend(loc=2)

# # X軸の目盛りを置換
# plt.xticks([1.15, 2.15, 3.15], label_x)
# plt.show()