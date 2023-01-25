import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

targetfinder_output_root = "https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder"
ep2vec_root = "https://github.com/wanwenzeng/ep2vec/raw/master"

data_root = os.path.join(os.path.dirname(__file__), "data")
fig_root = os.path.join(os.path.dirname(__file__), "figure")

def make_directory():
    # data_path = os.path.join(os.path.dirname(__file__), "data")
    # fig_path = os.path.join(os.path.dirname(__file__), "figure")

    os.system(f"mkdir -p {data_root}/TargetFinder")
    os.system(f"mkdir -p {data_root}/ep2vec")
    os.system(f"mkdir -p {data_root}/my")
    os.system(f"mkdir -p {fig_root}/TargetFinder")
    os.system(f"mkdir -p {fig_root}/ep2vec")
    os.system(f"mkdir -p {fig_root}/my")

def download_pair_data(cell_line):
    # # data directory を この~.pyと同じ場所に作成
    # data_path = os.path.join(os.path.dirname(__file__), "data")
    # os.system(f"mkdir -p {data_path}")

    # training data の url (TargetFinder)
    targetfinder_url = os.path.join(targetfinder_output_root, cell_line, "output-ep", "training.csv.gz")
    targetfinder_train_df = pd.read_csv(targetfinder_url,compression='gzip',error_bad_lines=False)

    # training data の url (ep2vec)
    ep2vec_url = os.path.join(ep2vec_root, f"{cell_line}train.csv")
    ep2vec_train_df = pd.read_csv(ep2vec_url)

    # 保存する名前 data下に置く
    targetfinder_training_data_filename = os.path.join(data_root, "TargetFinder", f"{cell_line}_train.csv")
    targetfinder_train_df.to_csv(targetfinder_training_data_filename, index=False)

    ep2vec_training_data_filename = os.path.join(data_root, "ep2vec", f"{cell_line}_train.csv")
    ep2vec_train_df.to_csv(ep2vec_training_data_filename, index=False)


def check_chr(cell_line):
    filename = os.path.join(os.path.dirname(__file__), "data", f"{cell_line}_train.csv")
    pair_df = pd.read_csv(filename, usecols=["enhancer_chrom","label","promoter_chrom"])



def check_PosNeg_ratio_by_each_promoter(cell_line, research_name_list):
    for research_name in research_name_list:
        train_df_path = os.path.join(data_root, research_name, f"{cell_line}_train.csv")
        pair_df = pd.read_csv(train_df_path, usecols=["enhancer_name","label","promoter_name"])

        positives = []
        negatives = []

        # 各promoter名毎に dataframe を分割し，sub-dataframeをループ全探索
        df_divided_by_promoters = pair_df.groupby(["promoter_name"])
        for promoter_name, df_by_promoter in df_divided_by_promoters:
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
        fig_path = os.path.join(fig_root, research_name, f"{cell_line}_pairs_by_each_promoter.png")
        figure.savefig(fig_path)


def check_PosNeg_ratio_by_each_enhancer(cell_line, research_name_list):
    for research_name in research_name_list:
        train_df_path = os.path.join(data_root, research_name, f"{cell_line}_train.csv")
        pair_df = pd.read_csv(train_df_path, usecols=["enhancer_name","label","enhancer_name"])

        positives = []
        negatives = []

        # 各enhancer名毎に dataframe を分割し，sub-dataframeをループ全探索
        df_divided_by_enhancers = pair_df.groupby(["enhancer_name"])
        for enhancer_name, df_by_enhancer in df_divided_by_enhancers:
            # print(enhancer_name)
            # print(f"pos : neg = {len(df_by_enhancer[df_by_enhancer['label'] == 1])} : {len(df_by_enhancer[df_by_enhancer['label'] == 0])}")
            pos = len(df_by_enhancer[df_by_enhancer['label'] == 1])
            neg = len(df_by_enhancer[df_by_enhancer['label'] == 0])
            positives.append(pos)
            negatives.append(neg)
        
        # 散布図を描画
        figure = plt.figure()
        plt.scatter(positives, negatives, alpha=0.3)
        plt.title(f"{cell_line} pos-neg cnt by each enhancer")
        plt.xlabel("positive cnt")
        plt.ylabel("negative cnt")
        # plt.show()
        fig_path = os.path.join(fig_root, research_name, f"{cell_line}_pairs_by_each_enhancer.png")
        figure.savefig(fig_path)




if __name__ == "__main__":
    make_directory()

    cell_line_list = ["K562", "GM12878", "HUVEC", "HeLa-S3", "NHEK"]
    # research_name_list = ["TargetFinder", "ep2vec", "my"]
    research_name_list = ["my"]
    for cell_line in cell_line_list:
        # download_pair_data(cell_line)
        check_PosNeg_ratio_by_each_promoter(cell_line, research_name_list)
        check_PosNeg_ratio_by_each_enhancer(cell_line, research_name_list)