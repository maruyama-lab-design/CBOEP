import pandas as pd
import requests
import os
import matplotlib.pyplot as plt

targetfinder_output_url = "https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder"

def make_directory():
    data_path = os.path.join(os.path.dirname(__file__), "data")
    fig_path = os.path.join(os.path.dirname(__file__), "figure")
    os.system(f"mkdir -p {data_path}")
    os.system(f"mkdir -p {fig_path}")

def download_pair_data(cell_line):
    # # data directory を この~.pyと同じ場所に作成
    # data_path = os.path.join(os.path.dirname(__file__), "data")
    # os.system(f"mkdir -p {data_path}")

    # training data の url
    url = os.path.join(targetfinder_output_url, cell_line, "output-ep", "training.csv.gz")
    pair_df = pd.read_csv(url,compression='gzip',error_bad_lines=False)

    # 保存する名前 data下に置く
    filename = os.path.join(os.path.dirname(__file__), "data", f"{cell_line}_train.csv")
    pair_df.to_csv(filename, index=False)


#____________...maybe unused...___________
def check_enhancer_pos_std():
    original_df = pd.read_csv("/Users/ylwrvr/卒論/Koga_code/check_targetfinder/pair_data/GM12878_train.csv", usecols=["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start"])
    pos_and_neg_df = original_df.groupby(["label"])
    for name, pos_or_neg_df in pos_and_neg_df:
        if name == 1:
            print("正例の場合")
        elif name == 0:
            print("不例の場合")
        df_divided_by_promoters = pos_or_neg_df.groupby(["promoter_name"])
        for promoter_name, df_by_promoter in df_divided_by_promoters:
            df_by_promoter["enhancer_center"] = (df_by_promoter["enhancer_start"] + df_by_promoter["enhancer_end"]) // 2
            print(f"{promoter_name}: cnt {len(df_by_promoter)} std {df_by_promoter['enhancer_center'].std()}")



def check_PosNeg_ratio_by_each_promoter(cell_line):
    csv_data_path = os.path.join(os.path.dirname(__file__), "data", f"{cell_line}_train.csv")
    pair_df = pd.read_csv(csv_data_path, usecols=["enhancer_name","label","promoter_name"])

    positives = []
    negatives = []

    # 各promoter名毎に dataframe を分割し，sub-dataframeをループ全探索
    df_divided_by_promoters = pair_df.groupby(["promoter_name"])
    for promoter_name, df_by_promoter in df_divided_by_promoters:
        # print(promoter_name)
        # print(f"pos : neg = {len(df_by_promoter[df_by_promoter['label'] == 1])} : {len(df_by_promoter[df_by_promoter['label'] == 0])}")
        pos = len(df_by_promoter[df_by_promoter['label'] == 1])
        neg = len(df_by_promoter[df_by_promoter['label'] == 0])
        positives.append(pos)
        negatives.append(neg)
    
    # 散布図を描画
    figure = plt.figure()
    plt.scatter(positives, negatives, alpha=0.3)
    plt.title(f"{cell_line} pos-neg cnt by each promoter")
    plt.xlabel("positive cnt")
    plt.ylabel("negative cnt")
    # plt.show()
    fig_path = os.path.join(os.path.dirname(__file__), "figure", f"{cell_line}_pairs_by_each_promoter.png")
    figure.savefig(fig_path)




if __name__ == "__main__":
    make_directory()

    cell_line_list = ["K562", "GM12878", "HUVEC", "HeLa-S3", "NHEK"]
    for cell_line in cell_line_list:
        download_pair_data(cell_line)
        check_PosNeg_ratio_by_each_promoter(cell_line)
