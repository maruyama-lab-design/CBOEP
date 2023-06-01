import pandas as pd


df = pd.read_csv("/Users/ylwrvr/卒論/Koga_code/main/pair_data/BENGI/cmn_test_pair/GM12878.csv")
df = df[df["enhancer_chrom"] == "chr1"]
pos_df = df[df["label"] == 1]
neg_df = df[df["label"] == 0]

for enh_name, subdf in pos_df.groupby("enhancer_name"):
    with open("/Users/ylwrvr/卒論/Koga_code/main/pair_data/BENGI/tmp/enhancer.bed", "w") as f:
        for _, data in subdf.iterrows():
            f.write(
                f"chr1\t{data['enhancer_start']}\t{data['enhancer_end']}\n"
            )
            break
    with open("/Users/ylwrvr/卒論/Koga_code/main/pair_data/BENGI/tmp/positive_promoter.bed", "w") as f:
        for _, data in subdf.iterrows():
            f.write(
                f"chr1\t{data['promoter_start']}\t{data['promoter_end']}\n"
            )
    with open("/Users/ylwrvr/卒論/Koga_code/main/pair_data/BENGI/tmp/negative_promoter.bed", "w") as f:
        for _, data in neg_df[neg_df["enhancer_name"] == enh_name].iterrows():
            if data["enhancer_distance_to_promoter"] > 999999999999:
                continue
            f.write(
                f"chr1\t{data['promoter_start']}\t{data['promoter_end']}\n"
            )
    break

        
