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


def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1],y_list[i-1])


def set_fontsize(axis_Max):
	x1, y1, x2, y2 = 100, 8, 625, 3
	a = (y2 - y1) / (x2 - x1)
	x = axis_Max * axis_Max
	x = max(x, 100)
	x = min(x, 625)
	y = a * (x - x1) + y1
	return y

def make_heatMap(args, chrom, VV, x_Max, y_Max, regionType, output_path):
	axis_Max = max(x_Max, y_Max)
	fontsize = set_fontsize(axis_Max)

	data = VV[:axis_Max+1, :axis_Max+1] # 二次元配列です

	mask = np.zeros_like(data)
	mask[np.where(data==0)] = True    

	plt.figure(figsize=(21, 14))
	fig, ax = plt.subplots()
	sns.heatmap(data, annot=True, square=True, annot_kws={"fontsize":fontsize}, fmt="d", cmap="Blues", linewidths=0.01, linecolor='black', mask=mask, cbar = False)
	ax.invert_yaxis()
	# ax.set_title(f"{args.cell_line} pos-neg cnt by each {regionType}")
	ax.set_xlabel("positive example count")
	ax.set_ylabel("negative example count")
	plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')


def datalist_to_dataframe(data_list):
	df_list = []
	for data in data_list:
		df_tmp = pd.read_table(data, header=None, index_col=None, names=["label", "distance", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "prm_chrom", "prm_start", "prm_end", "promoter_name"])
		df_list.append(df_tmp)
	df = pd.concat(df_list)
	print(df.head())
	return df
	

def get_df(args):
	data_list = glob.glob(os.path.join(os.path.dirname(__file__), args.data_name, args.data_type, f"{args.cell_line}*.tsv"))
	df = datalist_to_dataframe(data_list)
	return df



def make_PosNeg_figure(args):
	output_dir = os.path.join(os.path.dirname(__file__), "figure", args.data_name, args.data_type, args.cell_line)
	os.system(f"mkdir {output_dir}")

	df = get_df(args)

	for regionType in ["enhancer", "promoter"]:
		posCnt_Max = 0
		negCnt_Max = 0
		PosNeg_cnt = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
		columnName = regionType + "_name"

		for chrom, subdf in df.groupby("enhancer_chrom"):
			posCnt_Max_by_chrom = 0
			negCnt_Max_by_chrom = 0
			PosNeg_cnt_by_chrom = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
			for regionName, subsubdf in subdf.groupby(columnName):
				posCnt = len(subsubdf[subsubdf["label"] == 1])
				negCnt = len(subsubdf[subsubdf["label"] == 0])

				if posCnt > 20 or negCnt > 20:
					continue

				PosNeg_cnt_by_chrom[negCnt][posCnt] += 1

				posCnt_Max_by_chrom = max(posCnt_Max_by_chrom, posCnt)
				negCnt_Max_by_chrom = max(negCnt_Max_by_chrom, negCnt)

			output_path = os.path.join(output_dir, f"{chrom}_{regionType}.png")
			make_heatMap(args, chrom, PosNeg_cnt_by_chrom, posCnt_Max_by_chrom, negCnt_Max_by_chrom, regionType, output_path)
			
			PosNeg_cnt += PosNeg_cnt_by_chrom
			posCnt_Max = max(posCnt_Max, posCnt_Max_by_chrom)
			negCnt_Max = max(negCnt_Max, negCnt_Max_by_chrom)

		output_path = os.path.join(output_dir, f"chrAll_{regionType}.png")
		make_heatMap(args, chrom, PosNeg_cnt, posCnt_Max, negCnt_Max, regionType, output_path)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--data_name", help="", default="TargetFinder")
	parser.add_argument("--data_type", help="", default="original")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	args = parser.parse_args()

	# cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]

	for name in ["TargetFinder", "BENGI"]:
		for type in ["original", "maxflow"]:
			args.data_name = name
			args.data_type = type
			make_PosNeg_figure(args)

	# for researchName in research_list:
	# 	for cell_line in cell_line_list:
	# 		for ratio in ratio_list:
	# 			args.ratio = ratio
	# 			args.data_name = researchName
	# 			args.cell_line = cell_line
	# 			make_PosNeg_figure(args)

	# for cell_line in cell_line_list:
	# 	make_F1_barGraph(["TargetFinder", "ep2vec", "new"], cell_line, ["GBRT_100", "GBRT_1000", "GBRT_4000", "KNN_5", "KNN_10", "KNN_15"])
	# make_precision_recall_curve(["new", "TargetFinder", "ep2vec"], ["GBRT_100", "GBRT_1000", "GBRT_4000"])