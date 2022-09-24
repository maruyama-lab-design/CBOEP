import pandas as pd
import glob
import os
import random


def make_balanced_dataset(data_dir, out_dir):
    data_files = glob.glob(os.path.join(data_dir, "*.tsv"))
    for data_file in data_files:
        print(f"{data_file}..........")
        basename = os.path.basename(data_file)
        df = pd.read_table(data_file, header=None, names=["label", "_2", "_3", "_4", "_5", "enh_name", "_7", "_8", "_9", "prm_name"])
        pos_enh = {}
        pos_prm = {}
        pos_cnt = 0
        for _, data in df.iterrows():
            if data["label"] == 1:
                pos_cnt += 1
                pos_enh[data["enh_name"]] = 1
                pos_prm[data["prm_name"]] = 1

        neg_only_enh = {}
        neg_only_prm = {}
        neg_cnt = 0
        for _, data in df.iterrows():
            if data["label"] == 0:
                neg_cnt += 1
                if data["enh_name"] not in pos_enh:
                    neg_only_enh[data["enh_name"]] = 1
                if data["prm_name"] not in pos_prm:
                    neg_only_prm[data["prm_name"]] = 1

        print(f"positive pair: {pos_cnt}")
        print(f"negative pair: {neg_cnt}")
        print(f"negativeにしか現れないenhancer: {len(neg_only_enh)}個")
        print(f"negativeにしか現れないpromoter: {len(neg_only_prm)}個")


        diff = neg_cnt - pos_cnt
        diff = 999999999

        neg_only_enh = list(neg_only_enh.keys())
        neg_only_prm = list(neg_only_prm.keys())
        drop_enh_candidate_df = df[df['enh_name'].isin(neg_only_enh)]
        drop_prm_candidate_df = df[df['prm_name'].isin(neg_only_prm)]
        drop_index = random.sample(set(list(drop_enh_candidate_df.index) + list(drop_prm_candidate_df.index)), min(diff, len(set(list(drop_enh_candidate_df.index) + list(drop_prm_candidate_df.index)))))
        new_df = df.drop(drop_index)
        new_pos_cnt = len(new_df[new_df["label"] == 1])
        new_neg_cnt = len(new_df[new_df["label"] == 0])
        print(f"new positive pair: {new_pos_cnt}")
        print(f"new negative pair: {new_neg_cnt}")
        new_df.to_csv(os.path.join(out_dir, basename), index=False, header=False, sep="\t")

        print(f".........完了.........")




data_dir = "D:\ylwrv\Koga_code\TransEPI\data\BENGI\original"
out_dir = "D:\\ylwrv\\Koga_code\\TransEPI\\data\\BENGI\\constrained(all_regions_have_pos_pair)"
os.system(f"mkdir {out_dir}")
make_balanced_dataset(data_dir, out_dir)