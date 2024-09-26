from operator import index
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import argparse
import math
import glob

import collections

import math


def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1],y_list[i-1])


def set_fontsize(axis_Max):
	x1, y1, x2, y2 = 100, 9, 625, 2.5
	a = (y2 - y1) / (x2 - x1)
	x = axis_Max * axis_Max
	x = max(x, 100)
	x = min(x, 625)
	y = a * (x - x1) + y1
	return y

def make_heatMap(VV, size, outfile, regionType=""):
	fontsize = set_fontsize(size)

	data = VV[:size+1, :size+1] # 二次元配列です
	# data を
	# ・y = 2x より上
	# ・y = 0.5x より下
	# ・その間
	# で分ける
	data_upper = np.zeros_like(data)
	data_lower = np.zeros_like(data)
	data_middle = np.zeros_like(data)
	for j in range(size+1):
		for i in range(size+1):
			if j > 2 * i:
				data_upper[i][j] = data[i][j]
			elif j < 0.5 * i:
				data_lower[i][j] = data[i][j]
			else:
				data_middle[i][j] = data[i][j]



	mask_upper = np.zeros_like(data_upper)
	mask_upper[np.where(data_upper==0)] = True    
	mask_lower = np.zeros_like(data_lower)
	mask_lower[np.where(data_lower==0)] = True
	mask_middle = np.zeros_like(data_middle)
	mask_middle[np.where(data_middle==0)] = True

	plt.figure(figsize=(21, 21))
	fig, ax = plt.subplots()
	plt.plot([0, 48], [0, 24],color="gray", zorder=1, linestyle="dashed", linewidth=1)
	plt.plot([0, 30], [0, 60],color="gray", zorder=1, linestyle="dashed", linewidth=1)
	# sns.heatmap(
	# 	data, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"green"},
	# 	fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask, cbar = False, alpha=0
	# )
	sns.heatmap(
		data_upper, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"blue"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_upper, cbar = False, alpha=0
	)
	sns.heatmap(
		data_lower, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"red"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_lower, cbar = False, alpha=0
	)
	sns.heatmap(
		data_middle, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"green"},
		fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask_middle, cbar = False, alpha=0
	)
	for i in range(size+1): # 罫線
		plt.plot([i, i], [0, size+1], color="black", zorder=2, linewidth=0.1)
		plt.plot([0, size+1], [i, i], color="black", zorder=2, linewidth=0.1)

	ax.invert_yaxis()
	ax.set_xlabel(f"{regionType} freq in pos") # 先頭大文字
	ax.set_ylabel(f"{regionType} freq in neg")
	plt.setp(ax.get_xticklabels(), fontsize=6, rotation=0)
	plt.setp(ax.get_yticklabels(), fontsize=6)
	os.makedirs(os.path.dirname(outfile), exist_ok=True)
	plt.savefig(outfile, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')


def make_PosNeg_matrix(indir, outdir, cell, size=24):

	df = pd.read_csv(os.path.join(indir, f"{cell}.csv"))

	for regionType in ["enhancer", "promoter"]:

		PosNeg_cnt = np.zeros((1000, 1000), dtype="int64") # 大きめに用意

		for chrom, subdf in df.groupby("enhancer_chrom"):
			PosNeg_cnt_by_chrom = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
			for regionName, subsubdf in subdf.groupby(regionType + "_name"):
				posCnt = len(subsubdf[subsubdf["label"] == 1])
				negCnt = len(subsubdf[subsubdf["label"] == 0])

				PosNeg_cnt_by_chrom[negCnt][posCnt] += negCnt + posCnt
			
			PosNeg_cnt += PosNeg_cnt_by_chrom

			outfile = os.path.join(outdir,  f"{cell}_{regionType}.png")

		make_heatMap(PosNeg_cnt, size, outfile, regionType=regionType.capitalize())



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--indir", help="", default="TargetFinder")
	parser.add_argument("--outdir", help="")
	parser.add_argument("--cell", help="細胞株", default="GM12878")
	parser.add_argument("--size", type=int, default=24)
	args = parser.parse_args()

	cell7 = ["GM12878", "HeLa-S3", "HMEC", "HUVEC", "IMR90", "K562", "NHEK"]
	cell6 = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
	cell5 = ["GM12878", "HeLa-S3", "IMR90", "K562", "NHEK"]

	size = 10
	
	for cell in ["GM12878", "HeLa-S3", "HMEC", "HUVEC", "IMR90", "K562", "NHEK"]:
		if cell != "HUVEC":
			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "BENGI(org)")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000_region_restricted")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "BENGI(rm)")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000_region_restricted", "MCMC", "dmax_2500000,alpha_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBGS(BG)", "dmax_2500000,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000_region_restricted", "MCMC", "dmax_INF,alpha_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBGS(BG)", "dmax_INF,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)


			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000_region_restricted", "CBOEP", "dmax_2500000,p_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBMF(BG)", "dmax_2500000,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "BENGI_up_to_2500000_region_restricted", "CBOEP", "dmax_INF,p_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBMF(BG)", "dmax_INF,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

		if cell != "HMEC":

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "TargetFinder_up_to_2500000")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "TargetFinder")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)


			indir = os.path.join(os.path.dirname(__file__), "EPIs", "TargetFinder_up_to_2500000", "MCMC", "dmin_10000,dmax_2500000,alpha_1.0")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBGS(TF)", "dmax_2500000,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "TargetFinder_up_to_2500000", "MCMC", "dmin_10000,dmax_INF,alpha_1.0")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBGS(TF)", "dmax_INF,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "TargetFinder_up_to_2500000", "CBOEP", "dmin_10000,dmax_2500000,p_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBMF(TF)", "dmax_2500000,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

			indir = os.path.join(os.path.dirname(__file__), "EPIs", "TargetFinder_up_to_2500000", "CBOEP", "dmin_10000,dmax_INF,p_100")
			outdir = os.path.join(os.path.dirname(__file__), "P-N_matrix_10-10", "CBMF(TF)", "dmax_INF,alpha_100")
			os.makedirs(outdir, exist_ok=True)
			# if not os.path.exists(os.path.join(outdir, f"{cell}_enhancer.png")):
			make_PosNeg_matrix(indir, outdir, cell, size)

