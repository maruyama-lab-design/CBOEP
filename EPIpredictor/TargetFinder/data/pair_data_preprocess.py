from operator import index
import pandas as pd
import os
import glob


def download_from_TargetFinder():

    for cl in ["GM12878", "K562", "HeLa-S3", "IMR90", "NHEK"]:
        url = f"https://raw.githubusercontent.com/shwhalen/targetfinder/master/paper/targetfinder/{cl}/output-ep/pairs.csv"
        outpath = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", "TargetFinder", "original", f"{cl}.csv")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        os.system(f"wget {url} -O {outpath}")


def convert_regionName(name):
    # chr6:226523-228463|GM12878|EH37E0821432 to GM12878|chr1:9685722-9686400
    tmp = name.split("|")
    return tmp[1] + "|" + tmp[0]


def convert_BENGI_from_tsv2csv(dataname, datatype):

    for cl in ["GM12878", "K562", "HeLa", "IMR90", "NHEK", "HMEC"]:
        print(f"{cl}...")
        # files = glob.glob(os.path.join(os.path.dirname(__file__), "ep", "wo_feature", "BENGI", "original", f"{cl}*.tsv"))
        files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "TransEPI", "data", f"{dataname}", f"{datatype}", f"{cl}*.tsv"))
        for file in files:
            basename = os.path.splitext(os.path.basename(file))[0]
            print(f"{basename}...")
            outpath = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", f"{dataname}", f"{datatype}", f"{basename}.csv")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            columns = ["label", "enhancer_distance_to_promoter", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]
            df = pd.read_table(file, header=None, index_col=None, sep="\t", names=columns)
            df["enhancer_name"] = df["enhancer_name"].apply(convert_regionName)
            df["promoter_name"] = df["promoter_name"].apply(convert_regionName)
            print(f"size of duplicated data: {len(df[df.duplicated()])}")
            df = df[~df.duplicated()]
            assert df.duplicated().sum() == 0, df[df.duplicated()]
            df.to_csv(outpath, index=False)

    # url = f"https://github.com/biomed-AI/TransEPI/blob/main/data/BENGI/"
    # outpath = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", "BENGI")
    # os.system(f"wget {url} -p {outpath}")
    # # os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # return 0

    # for cl in ["GM12878", "K562", "HeLa-S3", "IMR90", "NHEK"]:
    #     url = f"https://github.com/biomed-AI/TransEPI/blob/main/data/BENGI/{cl}.HiC-Benchmark.v3.tsv.gz?raw=true"
    #     outpath = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", "TargetFinder", f"{cl}.csv")
    #     os.makedirs(os.path.dirname(outpath), exist_ok=True)
    #     os.system(f"wget {url} -O {outpath}")



def download_from_EP2vec():
    pass



# download_from_TargetFinder()
d_list = [d for d in list(range(2500000, 10000001, 500000)) + [9999999999]]
for dataname in ["BENGI", "TargetFinder"]:
    for datatype in [f"maxflow_{d}" for d in d_list] + ["original"]:
        convert_BENGI_from_tsv2csv(dataname, datatype)