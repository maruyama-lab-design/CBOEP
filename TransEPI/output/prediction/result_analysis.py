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


research_name_list = ["BG_org", "BG_mf"]

def result_analysis():
    column_names = ["data", "AUC", "AUPR", "F", "F\'", "balanced accuracy", "precison", "recall", "specificity", "NPV", "MCC"]
    data_list = []
    for research_name in research_name_list:
        for filename in glob.glob(f"one_cl\\{research_name}\\*prediction.txt"):
            rn = research_name + os.path.basename(filename).split(".")[0]
            # train, test = rn.split(",")
            # df = pd.read_table(filename, header=None, index_col=None, skiprows=2, names=["true", "prob", "chrom", "enh_name", "prm_name"])
            df = pd.read_table(filename, header=None, index_col=None, skiprows=1, names=["true", "prob"])
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
            
            data_list.append([rn, auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc])

    csv_df = pd.DataFrame(data=data_list, index=None, columns=column_names)
    with pd.ExcelWriter(os.path.join(__file__, "..", "TransEPI_result_analysis.xlsx"), mode="a", if_sheet_exists='replace') as writer:
        csv_df.to_excel(writer, sheet_name=f"one cell type")

        
        


result_analysis()