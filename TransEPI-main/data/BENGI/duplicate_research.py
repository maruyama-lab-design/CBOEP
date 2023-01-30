import pandas as pd
import os
import glob


def files2df(files):
    df = pd.DataFrame()
    for file in files:
        if len(df) == 0:
            df = pd.read_table(
                file,
                header=None,
                index_col=None,
                names=["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"]
            )
        else:
            tmp_df = pd.read_table(
                file,
                header=None,
                index_col=None,
                names=["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"]
            )
            df = pd.concat([df, tmp_df])
    return df


def check_duplicate_pair(cell_line):
    files = glob.glob(os.path.join(os.path.dirname(__file__), "data", "BENGI", f"{cell_line}*.tsv.gz"))
    df = files2df(files)
    df = df[df.duplicated(subset=['enh_start', "enh_end", "prm_start", "prm_end"], keep=False)]

    df["pair_name"] = df["enh_chrom"] + " : " + df["enh_start"].apply(str) + "-" + df["enh_end"].apply(str) + " <-> " + df["prm_start"].apply(str) + "-" + df["prm_end"].apply(str)

    df = df[["label", "pair_name", "enh_name", "prm_name", "enh_chrom", "prm_chrom"]]
    # df.to_csv("test.csv")

    analysis_list = [] # pair_name, pos_cnt, neg_cnt
    for pair_name, subdf in df.groupby("pair_name"):
        tmp_list = [pair_name, len(subdf[subdf["label"] == 1]), len(subdf[subdf["label"] == 0])]
        analysis_list.append(tmp_list)
    
    new_df = pd.DataFrame(data=analysis_list, columns=["pair name", "positive cnt", "negative cnt"])
    new_df.to_csv(f"{cell_line}-duplicated_pair.csv", index=False)


def count_duplicate_pair(cell_line):
    files = glob.glob(os.path.join(os.path.dirname(__file__), "data", "BENGI", f"{cell_line}*.tsv.gz"))
    df = files2df(files)
    df = df[df.duplicated(subset=['enh_start', "enh_end", "prm_start", "prm_end"], keep=False)]

    df["pair_name"] = df["enh_chrom"] + " : " + df["enh_start"].apply(str) + "-" + df["enh_end"].apply(str) + " <-> " + df["prm_start"].apply(str) + "-" + df["prm_end"].apply(str)

    df = df[["label", "pair_name", "enh_name", "prm_name", "enh_chrom", "prm_chrom"]]
    # df.to_csv("test.csv")

    analysis_list = [] # pair_name, pos_cnt, neg_cnt
    for pair_name, subdf in df.groupby("pair_name"):
        tmp_list = [pair_name, len(subdf[subdf["label"] == 1]), len(subdf[subdf["label"] == 0])]
        analysis_list.append(tmp_list)
    
    new_df = pd.DataFrame(data=analysis_list, columns=["pair name", "positive cnt", "negative cnt"])
    new_df.to_csv(f"{cell_line}-duplicated_pair.csv", index=False)


for cell_line in ["GM12878", "HeLa", "IMR90", "NHEK", "HMEC", "K562"]:
    check_duplicate_pair(cell_line)



