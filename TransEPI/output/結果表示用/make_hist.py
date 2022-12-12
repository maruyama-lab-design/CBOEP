import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, auc, f1_score

import glob




def make_hist(dataname, datatype):
    for filename in glob.glob(os.path.join(os.path.dirname(__file__), "result", f"{dataname}_{datatype}", "one_cl", "*.txt")):

        basename = os.path.splitext(os.path.basename(filename))[0]


        y_true = []
        y_pred = []

        result_csv = pd.read_table(filename)
        y_true += result_csv["true"].tolist()
        y_pred += result_csv["pred"].tolist()
        
        pred_t = []
        pred_f = []

        for (y_t, y_p) in zip(y_true, y_pred):
            if y_t == 1:
                pred_t.append(y_p)
            else:
                pred_f.append(y_p)

        bins = np.linspace(0, 1, 21)

        # plt.hist([pred_f, pred_t], bins, label=['F', 'T'])
        os.makedirs(os.path.join(os.path.dirname(__file__), "hist", f"{dataname}_{datatype}(one_cl)"), exist_ok=True)
        plt.figure()
        plt.hist(pred_f, bins, alpha=0.5, label='N', color="blue")
        plt.legend(loc='upper left')
        plt.title(dataname + "_" + datatype + " " + basename.split("_")[0])
        fig_path = os.path.join(os.path.dirname(__file__), "hist", f"{dataname}_{datatype}(one_cl)", basename + "_neg.png")
        plt.savefig(fig_path, dpi=300)

        plt.figure()
        plt.hist(pred_t, bins, alpha=0.5, label='P', color="red")
        plt.legend(loc='upper left')
        plt.title(dataname + "_" + datatype + " " + basename.split("_")[0])
        fig_path = os.path.join(os.path.dirname(__file__), "hist", f"{dataname}_{datatype}(one_cl)", basename + "_pos.png")
        plt.savefig(fig_path, dpi=300)


for dataname in ["BG"]:
    for datatype in ["org", "mf"]:
    # for datatype in ["org_mask-wn", "mf_mask-wn"]:
    # for datatype in ["org_useTest", "mf_useTest"]:
        make_hist(dataname, datatype)
        