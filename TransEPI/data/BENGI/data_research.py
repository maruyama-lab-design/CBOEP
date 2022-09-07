import pandas as pd
import glob

filenames = glob.glob("D:\\ylwrv\\TransEPI-main\\data\\BENGI\\*\\*.tsv")
for filename in filenames:
    df = pd.read_table(filename, header=None, names=["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"])
    distances = df["distance"].tolist()
    distances.sort(reverse=True)
    print(distances[0], distances[-1])