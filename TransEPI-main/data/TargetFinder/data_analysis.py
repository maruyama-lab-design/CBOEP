import pandas as pd
import glob
import os


def check_distance():
    filenames = glob.glob("D:\\ylwrv\\TransEPI-main\\data\\TargetFinder\\*\\*.tsv")
    for filename in filenames:
        df = pd.read_table(filename, header=None, names=["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"])
        distances = df["distance"].tolist()
        distances.sort(reverse=True)
        print(distances[0], distances[-1])


def data_analysis(excel_path, data_dir):
    column_names = ["filename", "enhancer", "promoter", "enh only used in positive", "prm only used in positive", "enh only used in negative", "prm only used in negative", "pos pair", "neg pair"]
    data_list = []

    filenames = glob.glob(os.path.join(data_dir, "*.tsv"))
    for filename in filenames:
        basename = os.path.basename(filename)
        df = pd.read_table(filename, header=None, index_col=None, names=["label", "distance", "e_chr", "enh_start", "enh_end", "enh_name", "p_chr", "prm_start", "prm_end", "prm_name"])
        pos_pair = len(df[df["label"] == 1])
        neg_pair = len(df[df["label"] == 0])

        enh = len(set(df["enh_name"].to_list()))
        prm = len(set(df["prm_name"].to_list()))

        pos_enh = {}
        pos_prm = {}
        pos_only_enh = {}
        pos_only_prm = {}
        for _, data in df.iterrows():
            if data["label"] == 1:
                pos_enh[data["enh_name"]] = 1
                pos_prm[data["prm_name"]] = 1

                pos_only_enh[data["enh_name"]] = 1
                pos_only_prm[data["prm_name"]] = 1

        neg_only_enh = {}
        neg_only_prm = {}
        for _, data in df.iterrows():
            if data["label"] == 0:
                if data["enh_name"] not in pos_enh:
                    neg_only_enh[data["enh_name"]] = 1
                if data["prm_name"] not in pos_prm:
                    neg_only_prm[data["prm_name"]] = 1

                if data["enh_name"] in pos_only_enh:
                    pos_only_enh.pop(data["enh_name"])
                if data["prm_name"] in pos_only_prm:
                    pos_only_prm.pop(data["prm_name"])

        enh_only_used_in_pos = len(pos_only_enh)
        prm_only_used_in_pos = len(pos_only_prm)
        enh_only_used_in_neg = len(neg_only_enh)
        prm_only_used_in_neg = len(neg_only_prm)

        data_list.append((basename, enh, prm, enh_only_used_in_pos, prm_only_used_in_pos, enh_only_used_in_neg, prm_only_used_in_neg, pos_pair, neg_pair))


    csv_df = pd.DataFrame(data=data_list, columns=column_names, index=None)
    csv_df.to_csv(os.path.join(data_dir, "TargetFinder_data_analysis.csv"), sep="\t", index=False)
    with pd.ExcelWriter(excel_path, mode="a", if_sheet_exists="replace") as writer:
        csv_df.to_excel(writer, sheet_name=f"TF_{os.path.basename(data_dir)}")


data_analysis("D:\ylwrv\Koga_code\TransEPI\data\TargetFinder\TargetFinder_data_analysis.xlsx", os.path.join(".", "constrained(all_regions_have_pos_pair)"))

    