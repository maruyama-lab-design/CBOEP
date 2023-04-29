import pandas as pd
import glob
import os



# BENGI と NIMF の負例での学習の質を比較するために共通にテストに用いる負例データを作成する
csv_columns = ["label", "distance", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]
test_chroms = ["chr19", "chr20", "chr21", "chr22", "chrX"]



def break_region_name(dataname, enhancer_name, promoter_name):
    enhancer_start, enhancer_end = map(int, enhancer_name.split(":")[1].split("|")[0].split("-"))
    promoter_start, promoter_end = map(int, promoter_name.split(":")[1].split("|")[0].split("-"))
    enhancer_pos = (enhancer_start + enhancer_end) / 2
    dist = abs(promoter_start - enhancer_pos)
    if dataname == "BENGI":
        promoter_start -= 1499
        promoter_end += 500
    return dist, enhancer_start, enhancer_end, promoter_start, promoter_end



def make_all_pair(dataname, cell_type):
    org_df = pd.read_csv(os.path.join(os.path.dirname(__file__), dataname, "original", f"{cell_type}.csv"))


    all_pair_dict = {
    }

    for enhancer_chrom, subdf in org_df.groupby("enhancer_chrom"):

        if enhancer_chrom not in ["chr19", "chr20", "chr21", "chr22", "chrX"]:
            continue
        if enhancer_chrom not in all_pair_dict.keys():
            all_pair_dict[enhancer_chrom] = []

        enhancer_list = set(subdf["enhancer_name"].to_list())
        promoter_list = set(subdf["promoter_name"].to_list())

        # print(f"enhancer: {len(enhancer_list)}")
        # print(f"promoter: {len(promoter_list)}")
        for enhancer_name in enhancer_list:
            for promoter_name in promoter_list:
                pair_name = f"{enhancer_name}={promoter_name}"
                all_pair_dict[enhancer_chrom].append(pair_name)


    data_list = []
    for chr in ["chr19", "chr20", "chr21", "chr22", "chrX"]: # 本当は全部でもよい
        all_pairs = all_pair_dict[chr]

        for pair in all_pairs:
            enhancer_name, promoter_name = pair.split("=")
            dist, enhancer_start, enhancer_end, promoter_start, promoter_end = break_region_name(dataname, enhancer_name, promoter_name)
            data_list.append([-999, dist, chr, enhancer_start, enhancer_end, enhancer_name, chr, promoter_start, promoter_end, promoter_name])

    all_pair_df = pd.DataFrame(data=data_list, columns=csv_columns)
    # print(f"all pair: {len(all_pair_df)}")

    org_df["pair"] = org_df["enhancer_name"] + "=" + org_df["promoter_name"]
    all_pair_df["pair"] = all_pair_df["enhancer_name"] + "=" + all_pair_df["promoter_name"]

    all_pair_df["label"] = 0 # 一旦全て負例
    all_pair_df.loc[all_pair_df["pair"].isin(org_df.loc[org_df["label"] == 1,'pair'].to_list()), "label"] = 1 # オリジナルで正例なら正例


    print(f"{cell_type} positive: {len(all_pair_df[all_pair_df['label'] == 1])}")
    print(f"{cell_type} negative: {len(all_pair_df[all_pair_df['label'] == 0])}")

    os.makedirs(os.path.join(os.path.dirname(__file__), dataname, "cmn_test_pair"), exist_ok=True)
    all_pair_df[csv_columns].to_csv(os.path.join(os.path.dirname(__file__), dataname, "cmn_test_pair", f"{cell_type}.csv"), index=False, sep=",")

   



for cell_type in ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]:
    for dataname in ["BENGI", "TargetFinder"]:
        make_all_pair(dataname, cell_type)
