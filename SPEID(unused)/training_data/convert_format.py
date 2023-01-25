import pandas as pd
import os
import glob


def convert_name(x):
    # chr1:24136556-24136600|GM12878 -> GM12878|chr1:9685722-9686400
    name, cl = x.split("|")[:2]
    new_x = cl + "|" + name
    return new_x


# TransEPIで使ったデータをSPEID用に変換する


dataname_list = ["TargetFinder", "BENGI"]
datatype_list = ["original", "maxflow", "constrained(all_regions_have_pos_pair)", "EP2vec"]

for dataname in dataname_list:
    for datatype in datatype_list:
        os.makedirs(os.path.join(os.path.dirname(__file__), dataname, datatype), exist_ok=True)
        filename_list = glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "TransEPI", "data", dataname, datatype, "*.tsv"))
        for filename in filename_list:
            basename = os.path.splitext(os.path.basename(filename))[0]
            outname = os.path.join(os.path.dirname(__file__), dataname, datatype, f"{basename}.csv")

            df = pd.read_table(filename, header=None, index_col=None, names=["label", "distance", "enhancer_chrom","enhancer_start","enhancer_end","enhancer_name","promoter_chrom","promoter_start","promoter_end","promoter_name"])
            df["enhancer_name"] = df["enhancer_name"].map(convert_name)
            df["promoter_name"] = df["promoter_name"].map(convert_name)

            df.to_csv(outname, sep=",", index=False)


 

 