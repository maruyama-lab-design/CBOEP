import result_analysis as ra
import os
import glob 


if __name__ == "__main__":
    train_cell = "GM12878"
    indirs = [
        # f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_{train_cell}",
        f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_region_restricted_{train_cell}",
        f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,p_100_{train_cell}",
        f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,alpha_100_{train_cell}"
    ]
    labels = [
        # "unnormalized BENGI",
        "BENGI",
        "CBMF",
        "CBGS"
    ]
    colors = [
        # "#bf7fff", # 紫
        "#00a7db", # 少し濃い青
        "#ff7f7f", # 赤
        "#00bb85" # 少し濃い緑
    ]
    marker_by_threshold = {
        0.05: "^", 0.5: "x", 0.95: "v",
    }
    ylim = (0.0, 1.0)
    folds = ["combined"]
    for test_cell in ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]:
        title = f"TransEPI test on {test_cell} all negatives"
        # title = ""
        
        # outfile = f"./PR-curve/cv-TransEPI_wo-mask_{test_cell}_all-neg(BGvsMFvsGS).png"
        outfile = f"./PR-curve/cv-TransEPI_wo-mask_{test_cell}_diff-neg(BGvsMFvsGS).png"
        ra.make_pr_curve(
            indirs=indirs,
            labels=labels,
            colors=colors,
            folds=folds,
            title=title,
            outfile=outfile,
            test_cell=test_cell,
            marker_by_threshold=marker_by_threshold,
            ylim=ylim
        )