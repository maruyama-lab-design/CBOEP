import pandas as pd


def reshape_for_TargetFinder(filename):
    df = pd.read_csv(filename)
    df["promoter_chrom"] = df["enhancer_chrom"]
    df["enhancer_start"] = df["enhancer_name"].apply(lambda x: int(x.split(":")[1].split("-")[0]))
    df["enhancer_end"] = df["enhancer_name"].apply(lambda x: int(x.split(":")[1].split("-")[1]))
    df["promoter_start"] = df["promoter_name"].apply(lambda x: int(x.split(":")[1].split("-")[0]))
    df["promoter_end"] = df["promoter_name"].apply(lambda x: int(x.split(":")[1].split("-")[1]))

    print(df.head())
    df.to_csv(filename, index=False)

reshape_for_TargetFinder("/Users/ylwrvr/卒論/Koga_code/training_data_research/training_data/new/×1/GM12878_train.csv")