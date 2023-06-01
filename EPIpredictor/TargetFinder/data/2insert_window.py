import pandas as pd
import glob
import os
import numpy as np


INF = 9999999999
def insert_window(dataname, datatype):
    cell_line_list = ["GM12878", "HeLa", "K562", "IMR90", "NHEK", "HMEC"]
    for cl in cell_line_list:
        print(f"{cl}...")
        files = glob.glob(os.path.join(os.path.dirname(__file__), "ep", "wo_feature", dataname, datatype, f"{cl}*.csv"))
        for file in files:
            outpath = os.path.join(os.path.dirname(__file__), "epw", "wo_feature", dataname, datatype, os.path.basename(file))
            os.makedirs(os.path.dirname(outpath), exist_ok=True)

            df = pd.read_csv(file, sep=",")
            df["window_chrom"] = df["enhancer_chrom"]
            start = []
            end = []
            name = []


            df["window_start"] = np.where(df["enhancer_end"] < df["promoter_end"], df["enhancer_end"], df["promoter_end"])
            df["window_end"] = np.where(df["enhancer_start"] > df["promoter_start"], df["enhancer_start"], df["promoter_start"])
            df["window_name"] = cl + "|" + df["window_chrom"] + ":" + df["window_start"].apply(str) + "-" +  df["window_end"].apply(str)


            assert (df["window" + '_end'] >= df["window" + '_start']).all(), df[df["window" + '_end'] <= df["window" + '_start']].head()

            df.to_csv(outpath, index=False)



d_list = [d for d in list(range(2500000, 10000001, 500000)) + [9999999999]]
datatype_list = [f"maxflow_{d}" for d in d_list] + ["original"]
for dataname in ["BENGI", "TargetFinder"]:
    for datatype in datatype_list:
        print(f"{dataname}-{datatype}...")
        insert_window(dataname, datatype)