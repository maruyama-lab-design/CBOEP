import result_analysis as ra
import os
import glob



if __name__ == "__main__":

    os.makedirs(os.path.dirname(f"./bar/EPIs-score/BG(rm)vsCBMF(BG)vsCBGS(BG).png"), exist_ok=True)
    ra.make_dataset_score_bar(
        indirs=[
            # f"./EPIs/BENGI_up_to_2500000",
            f"./EPIs/BENGI_up_to_2500000_region_restricted",
            f"./EPIs/BENGI_up_to_2500000_region_restricted/CBOEP/dmax_2500000,p_100",
            f"./EPIs/BENGI_up_to_2500000_region_restricted/MCMC/dmax_2500000,alpha_100"
        ],
        labels=[
            # "unnormalized BENGI",
            "BENGI-neg",
            "CBMF(BG)-neg",
            "CBGS(BG)-neg"
        ],
        colors=[
        # "#bf7fff", # 紫
        "#7fbfff", # 青
        "#ff7f7f", # 赤
        "#7fff7f" # 緑
        ],
        ylim=(0.0, 8.0),
        outfile=f"./bar/EPIs-score/BG(rm)vsCBMF(BG)vsCBGS(BG).png",
    )

    os.makedirs(os.path.dirname(f"./bar/EPIs-score/TFvsCBMF(TF)vsCBGS(TF).png"), exist_ok=True)
    ra.make_dataset_score_bar(
        indirs=[
            f"./EPIs/TargetFinder_up_to_2500000",
            f"./EPIs/TargetFinder_up_to_2500000/CBOEP/dmin_10000,dmax_2500000,p_100",
            f"./EPIs/TargetFinder_up_to_2500000/MCMC/dmin_10000,dmax_2500000,alpha_1.0"
        ],
        labels=[
            "TargetFinder-neg",
            "CBMF(TF)-neg",
            "CBGS(TF)-neg"
        ],
        colors=[
            "#bf7fff", # 紫
            "#ff7f7f", # 赤
            "#7fff7f" # 緑
        ],
        ylim=(0.0, 4.0),
        cells = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"],
        outfile=f"./bar/EPIs-score/TFvsCBMF(TF)vsCBGS(TF).png",
    )