import result_analysis as ra
import os
import glob


if __name__ == "__main__":

    train_cell = "GM12878"
    legend = {
        "fontsize": 8,
        "loc": "upper center",
        "ncol": 3
    }
    indirs_TransEPI_BG = [
        f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_{train_cell}",
        f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_region_restricted_{train_cell}",
        # f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,p_100_{train_cell}",
        # f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,alpha_100_{train_cell}"
    ]
    indirs_TransEPI_TF = [
        f"./TransEPI/2023-01-16_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_TargetFinder_up_to_2500000_{train_cell}",
        f"./TransEPI/2023-01-16_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmin_10000,dmax_2500000,p_100_{train_cell}",
        f"./TransEPI/2023-01-16_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmin_10000,dmax_2500000,alpha_1.0_{train_cell}"
    ]
    indirs_TransEPI_all_neg = [
        f"./TransEPI/common_test/BENGI_up_to_2500000_region_restricted",
        f"./TransEPI/common_test/CBMF",
        f"./TransEPI/common_test/CBGS"
    ]
    indirs_TargetFinder_BG = [
        f"./TargetFinder/prediction_w-window/BENGI_up_to_2500000/train_{train_cell}",
        f"./TargetFinder/prediction_w-window/BENGI_up_to_2500000_region_restricted/train_{train_cell}",
        f"./TargetFinder/prediction_w-window/CBOEP(dmax_2500000,alpha_100)/train_{train_cell}",
        f"./TargetFinder/prediction_w-window/MCMC(dmax_2500000,alpha_100)/train_{train_cell}"
    ]
    indirs_TargetFinder_TF = [
        f"./TargetFinder/prediction_w-window/TargetFinder_up_to_2500000/train_{train_cell}",
        f"./TargetFinder/prediction_w-window/TargetFinder_up_to_2500000/CBOEP/dmin_10000,dmax_2500000,p_100/train_{train_cell}",
        f"./TargetFinder/prediction_w-window/TargetFinder_up_to_2500000/MCMC/dmin_10000,dmax_2500000,alpha_1.0/train_{train_cell}"
    ]

    labels_BG = [
        # "unnormalized BENGI",
        "BENGI",
        "CBMF",
        "CBGS"
    ]

    labels_TF = [
        "TargetFinder",
        "CBMF(TF)",
        "CBGS(TF)"
    ]

    colors=[
        # "#bf7fff", # 紫
        "#7fbfff", # 青
        "#ff7f7f", # 赤
        "#7fff7f" # 緑
    ]

    cell6 = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
    # cell_TF = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
    cell5 = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]
    # for metric in ["Balanced accuracy", "Recall", "Specificity", "AUC", "AUPR", "AUPR-ratio", "F1", "MCC", "precision"]:
    for metric in ["Balanced accuracy", "Specificity", "Recall", "AUPR"]:

        outfile_TransEPI_all_neg = f"./bar/{metric.replace(' ', '-')}/cv-TransEPI_wo-mask_all-neg(BGvsMFvsGS).png"
        os.makedirs(os.path.dirname(outfile_TransEPI_all_neg), exist_ok=True)
        ra.make_result_bar(
            indirs=indirs_TransEPI_all_neg,
            labels=labels_BG,
            colors=colors,
            outfile=outfile_TransEPI_all_neg,
            metric=metric,
            test_cells=cell6,  
            n_fold=5,
            legend=legend
        )
        continue

        # 空白を含まないmetric名に変換
        # outfile_TransEPI = f"./bar/{metric.replace(' ', '-')}/cv-TransEPI_wo-mask_BG(org)vsBG(rm)vsMFvsGS.png"
        # outfile_TransEPI_BG = f"./bar/{metric.replace(' ', '-')}/cv-TransEPI_wo-mask_BG(org)vsBG(rm).png"
        # os.makedirs(os.path.dirname(outfile_TransEPI_BG), exist_ok=True)

        # outfile_TargetFinder = f"./bar/{metric.replace(' ', '-')}/cv-TargetFinder_wo-mask_BG(org)vsBG(rm)vsMFvsGS.png"
        # outfile_TargetFinder_BG = f"./bar/{metric.replace(' ', '-')}/cv-TargetFinder_wo-mask_BG(org)vsBG(rm).png"
        # os.makedirs(os.path.dirname(outfile_TargetFinder_BG), exist_ok=True)
       
        outfile_TransEPI_TF = f"./bar/{metric.replace(' ', '-')}/cv-TransEPI_wo-mask_TFvsMF(TF)vsGS(TF).png"
        os.makedirs(os.path.dirname(outfile_TransEPI_TF), exist_ok=True)


        outfile_TargetFinder_TF = f"./bar/{metric.replace(' ', '-')}/cv-TargetFinder_wo-mask_TFvsMF(TF)vsGS(TF).png"
        os.makedirs(os.path.dirname(outfile_TargetFinder_TF), exist_ok=True)

        # ra.make_result_bar(
        #     indirs=indirs_TransEPI_TF,
        #     labels=labels_TF, colors=colors,
        #     outfile=outfile_TransEPI_TF,
        #     metric=metric,
        #     test_cells=cell5,
        #     n_fold=5,legend=legend
        # )

        ra.make_result_bar(
            indirs=indirs_TargetFinder_TF,
            labels=labels_TF, colors=colors,
            outfile=outfile_TargetFinder_TF,
            metric=metric,
            test_cells=["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"],
            n_fold=5,legend=legend
        )



        # ra.make_result_bar(
        #     indirs=indirs_TransEPI_BG,
        #     labels=labels,
        #     colors=colors,
        #     outfile=outfile_TransEPI_BG,
        #     metric=metric,
        #     test_cells=cell6,
        #     n_fold=5,
        #     legend=legend
        # )
        
        # ra.make_result_bar(
        #     indirs=indirs_TargetFinder_BG,
        #     labels=labels,
        #     colors=colors,
        #     outfile=outfile_TargetFinder_BG,
        #     metric=metric,
        #     test_cells=cell5,
        #     n_fold=5,
        #     legend=legend
        # )



        continue

        ra.make_reuslt_bar(
            indirs=[
                f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_region_restricted_{train_cell}",
                f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_dmax_2500000,p_100_{train_cell}",
                f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_dmax_2500000,alpha_100_{train_cell}"
            ],
            labels=[
                "BENGI-negative",
                "CBMF-negative",
                "CBGS-negative"
            ],
            colors=[
                "#7fbfff", # 鮮やかな赤
                "#ff7f7f", # 鮮やかな青
                "#7fff7f" # 重厚な青みの緑
            ],
            outfile=f"./bar/{metric}/cv-TransEPI_w-mask_BGvsMFvsGS.png",
            metric=metric,
            test_cells=["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"],
            n_fold=5
        )

        ra.make_reuslt_bar(
            indirs=[
                f"./TargetFinder/prediction_w-window/BENGI_up_to_2500000_region_restricted/train_{train_cell}",
                f"./TargetFinder/prediction_w-window/CBOEP(dmax_2500000,alpha_100)/train_{train_cell}",
                f"./TargetFinder/prediction_w-window/MCMC(dmax_2500000,alpha_100)/train_{train_cell}"
            ],
            labels=[
                "BENGI-negative",
                "CBMF-negative",
                "CBGS-negative"
            ],
            colors=[
                "#7fbfff", # 鮮やかな赤
                "#ff7f7f", # 鮮やかな青
                "#7fff7f" # 重厚な青みの緑
            ],
            outfile=f"./bar/{metric}/cv-TargetFinder_wo-mask_BGvsMFvsGS.png",
            metric=metric,
            test_cells=["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"],
            n_fold=5
        )