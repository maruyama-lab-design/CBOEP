from email import header
import pandas as pd
import os
import glob
BENGI_columns = ["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"]

def convert_TargetFinder_to_BENGI_format(filename):
    csv_df = pd.read_csv(filename)

    data = []

    # distance col が無いので作る

    for _, row in csv_df.iterrows():
        label = row["label"] 
        distance = max(row["enhancer_start"] - row["promoter_end"], row["promoter_start"] - row["enhancer_end"])
        chrom = row["enhancer_chrom"] 
        enh_s = row["enhancer_start"] 
        enh_e = row["enhancer_end"] 
        enh_n = row["enhancer_name"] 
        prm_s = row["promoter_start"] 
        prm_e = row["promoter_end"] 
        prm_n = row["promoter_name"] 
    
        data.append([label, distance, chrom, enh_s, enh_e, enh_n, chrom, prm_s, prm_e, prm_n])

    tsv_df = pd.DataFrame(data=data)
    outname = os.path.splitext(filename)[0] + ".tsv"
    tsv_df.to_csv(outname, header=False, index=False, sep="\t")


for filename in glob.glob(os.path.join(__file__, "..", "original", "*.csv")):
    convert_TargetFinder_to_BENGI_format(filename)

