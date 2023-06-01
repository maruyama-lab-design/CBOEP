import pandas as pd
import numpy as np
import os


# positive EPI count by chromosome 
def check_EPIcnt_by_chrom(dataname="BENGI"):

    if dataname == "BENGI":
        cell_line_list = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
    elif dataname == "TargetFinder":
        cell_line_list = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
    chrom_list = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
    data_list = [[0 for i in range(len(cell_line_list))] for j in range(23)]
    for i, cell_line in enumerate(cell_line_list):

        all_data = pd.read_csv(os.path.join(os.path.dirname(__file__), dataname, "original", f"{cell_line}.csv"), usecols=["label", "enhancer_chrom"])


        for chrom, df_by_chrom in all_data.groupby("enhancer_chrom"):
            if chrom.split("r")[-1] == "X":
                index = 22
            else:
                index = int(chrom.split("r")[-1]) - 1

            data_list[index][i] = len(df_by_chrom[df_by_chrom["label"] == 1])

    analysis_df = pd.DataFrame(data=data_list, index=chrom_list, columns=cell_line_list)
    
    outdir = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(outdir, exist_ok=True)

    analysis_df.to_csv(os.path.join(outdir, f"{dataname}_EPIcnt.csv"))
    print(analysis_df)




def check_neg_EPIcnt_by_chrom_in_cmn_test(dataname="BENGI"):

    if dataname == "BENGI":
        cell_line_list = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
    elif dataname == "TargetFinder":
        cell_line_list = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
    chrom_list = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
    data_list = [[0 for i in range(len(cell_line_list))] for j in range(23)]
    for i, cell_line in enumerate(cell_line_list):

        all_data = pd.read_csv(os.path.join(os.path.dirname(__file__), dataname, "cmn_test_pair", f"{cell_line}.csv"), usecols=["label", "enhancer_chrom"])


        for chrom, df_by_chrom in all_data.groupby("enhancer_chrom"):
            if chrom.split("r")[-1] == "X":
                index = 22
            else:
                index = int(chrom.split("r")[-1]) - 1

            data_list[index][i] = len(df_by_chrom[df_by_chrom["label"] == 1])

    analysis_df = pd.DataFrame(data=data_list, index=chrom_list, columns=cell_line_list)
    
    outdir = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(outdir, exist_ok=True)

    analysis_df.to_csv(os.path.join(outdir, f"{dataname}_cmn_posEPIcnt.csv"))
    print(analysis_df)


# check_EPIcnt_by_chrom(dataname="BENGI")
# check_EPIcnt_by_chrom(dataname="TargetFinder")
check_neg_EPIcnt_by_chrom_in_cmn_test()
