from email import header
import pandas as pd
import os
import glob
BENGI_columns = ["label", "distance", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]

def convert_TargetFinder_to_BENGI_format(filename):
    # csv_df = pd.read_table(filename, header=None, index_col=None, names=BENGI_columns)
    csv_df = pd.read_csv(filename, index_col=None)

    data = []

    # distance col が無いので作る

    for _, row in csv_df.iterrows():
        label = row["label"] 
        distance = max(row["enhancer_start"] - row["promoter_end"], row["promoter_start"] - row["enhancer_end"])
        chrom = row["enhancer_chrom"] 
        enh_s = row["enhancer_start"] 
        enh_e = row["enhancer_end"] 
        prm_s = row["promoter_start"] 
        prm_e = row["promoter_end"] 

        # name = chr1:XXXXXXX-YYYYYYY|GM12878|ESC----
        cell_line, enh_n = row["enhancer_name"].split("|")
        enh_n = enh_n + "|" + cell_line

        cell_line, prm_n = row["promoter_name"].split("|")
        prm_n = prm_n + "|" + cell_line
    
        data.append([label, distance, chrom, enh_s, enh_e, enh_n, chrom, prm_s, prm_e, prm_n])

    tsv_df = pd.DataFrame(data=data)
    outname = os.path.splitext(filename)[0] + ".tsv"
    tsv_df.to_csv(outname, header=False, index=False, sep="\t")


for filename in glob.glob(os.path.join(__file__, "..", "EP2vec", "*.csv")):
    convert_TargetFinder_to_BENGI_format(filename)

