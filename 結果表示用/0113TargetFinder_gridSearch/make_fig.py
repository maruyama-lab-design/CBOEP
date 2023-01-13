import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt




def make_hist(dataname, datatype, train_cl, test_cl):
    filenames = glob.glob(os.path.join(os.path.dirname(__file__), "result", "cell_type_wise(epw)", f"{dataname}_{datatype}", f"{train_cl}-{test_cl}*.csv"))
    for filename in filenames:
        outname = os.path.join(os.path.dirname(__file__), "fig", "hist", "cell_type_wise(epw)", f"{dataname}_{datatype}", os.path.basename(filename).replace(".csv", ".png"))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        df = pd.read_csv(filename)
        y_true = df["y_test"].tolist()
        y_pred = df["y_pred"].tolist()

        pred_t = []
        pred_f = []

        for (y_t, y_p) in zip(y_true, y_pred):
            if y_t == 1:
                pred_t.append(y_p)
            else:
                pred_f.append(y_p)

        bins = np.linspace(0, 1, 21)

        # plt.hist([pred_f, pred_t], bins, label=['F', 'T'])
        plt.figure()
        plt.hist(pred_f, bins, alpha=0.5, label='N')
        plt.hist(pred_t, bins, alpha=0.5, label='P')
        plt.legend(loc='upper left')
        # plt.ylim((0, 250))
        plt.savefig(outname, dpi=300)




for dataname in ["TargetFinder"]:
    for datatype in ["original", "maxflow_10000000"]:
        for train_cl in ["GM12878"]:
            for test_cl in ["GM12878", "HeLa-S3", "K562"]:
                make_hist(dataname, datatype, train_cl, test_cl)