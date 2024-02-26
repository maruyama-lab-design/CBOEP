import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt



# 負例のみに現れるエンハンサーまたはプロモーターを含むEPIの正答率を求める
def check_only_negative_EPI_accuracy(
        indir = os.path.join(
            os.path.dirname(__file__),
            "TransEPI", "2023-11-01_cv_TransEPI", "wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_GM12878"
        ),
        test_cell = "GM12878",
        folds = "combined",
        outfile = ""
):
    
    print(f"___ Test cell: {test_cell}___ ")

    # データの読み込み
    df = pd.DataFrame()
    if folds == "combined":
        for fold in range(1, 6):
            if len(glob.glob(os.path.join(indir, f"*{test_cell}_fold{fold}.txt"))) == 0:
                if len(glob.glob(os.path.join(indir, f"*{test_cell}_fold{fold}.csv"))) == 1:
                    infile = glob.glob(os.path.join(indir, f"*{test_cell}_fold{fold}.csv"))[0]
                else:
                    raise ValueError("There are multiple files.")
            elif len(glob.glob(os.path.join(indir, f"*{test_cell}_fold{fold}.txt"))) == 1:
                infile = glob.glob(os.path.join(indir, f"*{test_cell}_fold{fold}.txt"))[0]
            else:
                raise ValueError("There are multiple files.")
            df = pd.concat([
                df,
                pd.read_csv(infile, sep="\t")
            ])

    print(f"Actual Positive: {len(df[df['true'] == 1])}")
    print(f"Actual Negative: {len(df[df['true'] == 0])}")

    # 正例に現れるエンハンサーとプロモーターの集合を取得
    positive_enhancer_set = set(df[df["true"] == 1]["enhancer_name"])
    positive_promoter_set = set(df[df["true"] == 1]["promoter_name"])
    
    # 負例に現れるエンハンサーとプロモーターの集合を取得
    negative_enhancer_set = set(df[df["true"] == 0]["enhancer_name"])
    negative_promoter_set = set(df[df["true"] == 0]["promoter_name"])

    # 負例にのみ現れるエンハンサーとプロモーターの集合を取得
    only_negative_enhancer_set = negative_enhancer_set - positive_enhancer_set
    only_negative_promoter_set = negative_promoter_set - positive_promoter_set

    # 負例にのみ現れるエンハンサーとプロモーターを含むnegative EPIを取得
    only_negative_EPI = df[
        (df["enhancer_name"].isin(only_negative_enhancer_set)) | (df["promoter_name"].isin(only_negative_promoter_set))
    ]
    print(f"Only negative EPI: {len(only_negative_EPI)}")

    # 正例にも現れているエンハンサーとプロモーターを含むnegative EPIを取得
    negative_EPI = df[
        (df["enhancer_name"].isin(positive_enhancer_set)) & (df["promoter_name"].isin(positive_promoter_set))
        & (df["true"] == 0)
    ]
    print(f"normalized BG negative EPI: {len(negative_EPI)}")

    # 負例にのみ現れるエンハンサーとプロモーターを含むEPIの正答率を求める
    only_negative_EPI_accuracy = len(only_negative_EPI[only_negative_EPI["pred"] < 0.5]) / len(only_negative_EPI)
    print(f"Only negative EPI accuracy: {len(only_negative_EPI[only_negative_EPI['pred'] < 0.5])} / {len(only_negative_EPI)} = {only_negative_EPI_accuracy}")

    # 正例にも現れているエンハンサーとプロモーターを含むnegative EPIの正答率を求める
    negative_EPI_accuracy = len(negative_EPI[negative_EPI["pred"] < 0.5]) / len(negative_EPI)
    print(f"normalized BG negative EPI accuracy: {len(negative_EPI[negative_EPI['pred'] < 0.5])} / {len(negative_EPI)} = {negative_EPI_accuracy}")


    print("___ End of test ___\n\n")


    plt.figure()
    x = [-1, 1]
    # 負例にのみ現れるエンハンサーとプロモーターを含むnegative EPIの個数と
    # 正例にも現れているエンハンサーとプロモーターを含むnegative EPIの個数を棒グラフで表示
    plt.bar(
        x=x,
        height=[len(only_negative_EPI), len(negative_EPI)],
        width=0.5,
        color=["blue", "blue"],
        fill=False,
        edgecolor="blue"
    )

    # 負例にのみ現れるエンハンサーとプロモーターを含むnegative EPIの正答率と
    # 正例にも現れているエンハンサーとプロモーターを含むnegative EPIの正答率を棒グラフで表示
    plt.bar(
        x=x,
        height=[len(only_negative_EPI[only_negative_EPI['pred'] < 0.5]), len(negative_EPI[negative_EPI['pred'] < 0.5])],
        width=0.5,
        color=["blue", "blue"],
        alpha=0.5
    )
    # 棒グラフの横に正答率を表示
    plt.text(
        x=x[0],
        y=len(only_negative_EPI)*1.001,
        s=f"{only_negative_EPI_accuracy*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )
    plt.text(
        x=x[1],
        y=len(negative_EPI)*1.001,
        s=f"{negative_EPI_accuracy*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

    plt.grid(axis="y", linestyle="dotted", color="gray", linewidth=0.4)
    # plt.xticks(x, [r"$\alpha$", r"$\beta$"])
    plt.xticks(x, ["", ""])
    plt.xlim(-2, 2)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    # plt.show()
    plt.savefig(outfile, dpi=300)



if __name__ == "__main__":

    cell6 = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
    cell5 = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]

    indir_TransEPI = os.path.join(
        os.path.dirname(__file__),
        "TransEPI", "2023-11-01_cv_TransEPI", "wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_GM12878"
    )

    indir_TargetFinder = os.path.join(
        os.path.dirname(__file__),
        "TargetFinder", "prediction_w-window", "BENGI_up_to_2500000/train_GM12878"
    )



    for test_cell in cell6:
        outfile_TransEPI = os.path.join(
            os.path.dirname(__file__),
            "bar",
            "raw-BG_accuracy",
            f"TransEPI-{test_cell}.png"
        )
        os.makedirs(os.path.dirname(outfile_TransEPI), exist_ok=True)

        outfile_TargetFinder = os.path.join(
            os.path.dirname(__file__),
            "bar",
            "raw-BG_accuracy",
            f"TargetFinder-{test_cell}.png"
        )
        os.makedirs(os.path.dirname(outfile_TargetFinder), exist_ok=True)
        

        check_only_negative_EPI_accuracy(
            indir=indir_TransEPI,
            test_cell=test_cell,
            outfile=outfile_TransEPI
        )

        
        if test_cell == "HMEC":
            continue
        check_only_negative_EPI_accuracy(
            indir=indir_TargetFinder,
            test_cell=test_cell,
            outfile=outfile_TargetFinder
        )
        
    