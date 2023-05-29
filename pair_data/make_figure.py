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

def make_heatMap(args, VV, x_Max, y_Max, regionType, output_path):
	axis_Max = max(x_Max, y_Max)
	fontsize = set_fontsize(axis_Max)

	data = VV[:axis_Max+1, :axis_Max+1] # 二次元配列です

	mask = np.zeros_like(data)
	mask[np.where(data==0)] = True    

	plt.figure(figsize=(21, 21))
	fig, ax = plt.subplots()
	plt.plot([0, 48], [0, 24],color="gray", zorder=1, linestyle="dashed", linewidth=1)
	plt.plot([0, 30], [0, 60],color="gray", zorder=1, linestyle="dashed", linewidth=1)
	sns.heatmap(data, annot=True, square=True, annot_kws={"fontsize":fontsize, "color":"green"}, fmt="d", cmap="Blues", linewidths=0.1, linecolor='black', mask=mask, cbar = False, alpha=0)
	for i in range(26):
		plt.plot([i, i], [0, 26], color="black", zorder=2, linewidth=0.1)
		plt.plot([0, 26], [i, i], color="black", zorder=2, linewidth=0.1)

	ax.invert_yaxis()
	ax.set_xlabel("positive example count")
	ax.set_ylabel("negative example count")
	plt.setp(ax.get_xticklabels(), fontsize=5, rotation=0)
	plt.setp(ax.get_yticklabels(), fontsize=5)
	plt.savefig(output_path, format="png", dpi=1000, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')


def make_PosNeg_matrix(args):
	output_dir = os.path.join(os.path.dirname(__file__), "figure", "matrix")
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


				# PosNeg_cnt_by_chrom[negCnt][posCnt] += 1
				PosNeg_cnt_by_chrom[negCnt][posCnt] += negCnt + posCnt

				posCnt_Max_by_chrom = max(posCnt_Max_by_chrom, posCnt)
				negCnt_Max_by_chrom = max(negCnt_Max_by_chrom, negCnt)

			# output_path = os.path.join(output_dir, f"{chrom}_{regionType}.png")
			# make_heatMap(args, chrom, PosNeg_cnt_by_chrom, 24, 24, regionType, output_path)
			
			PosNeg_cnt += PosNeg_cnt_by_chrom
			# posCnt_Max = max(posCnt_Max, posCnt_Max_by_chrom)
			# negCnt_Max = max(negCnt_Max, negCnt_Max_by_chrom)

		output_path = os.path.join(output_dir,  f"{args.data_name}_{args.data_type}_{args.cell_line}_{regionType}.png")
		make_heatMap(args, PosNeg_cnt, 24, 24, regionType, output_path)


def datalist_to_dataframe(data_list):
	df_list = []
	for data in data_list:
		df_tmp = pd.read_table(data, header=None, index_col=None, names=["label", "distance", "enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "prm_chrom", "prm_start", "prm_end", "promoter_name"])
		df_list.append(df_tmp)
	df = pd.concat(df_list)
	# print(df.head())
	return df
	

def get_df(args):
	dfs = []
	if not args.data_types is None:
		for data_type in args.data_types:
			filename = os.path.join(os.path.dirname(__file__), args.data_name, data_type, f"{args.cell_line}.csv")
			df = pd.read_csv(filename, usecols=["label", "enhancer_chrom", "enhancer_name", "promoter_name"])
			dfs.append(df)
		return dfs
	else:
		filename = os.path.join(os.path.dirname(__file__), args.data_name, args.data_type, f"{args.cell_line}.csv")
		df = pd.read_csv(filename, usecols=["label", "enhancer_chrom", "enhancer_name", "promoter_name"])
		return df



def classify_into_bin(x, bin_size):
	if x > 0:
		return math.ceil(x / bin_size) * bin_size
	elif x < 0:
		return math.floor(x / bin_size) * bin_size
	else:
		return 0



def make_positive_negative_gap_ratio_fig(args, positive_negative_gap_ratio_by_df, bin_size, regionType):

	output_dir = os.path.join(os.path.dirname(__file__), "figure")
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, f"{args.data_name}_{args.cell_line}_{regionType}_{bin_size}.png")
	plt.figure()
	for i, positive_negative_gap_ratio in enumerate(positive_negative_gap_ratio_by_df):
		positive_negative_gap_ratio = list(map(lambda x: classify_into_bin(x, bin_size), positive_negative_gap_ratio))
		c = collections.Counter(positive_negative_gap_ratio)
		n = math.ceil(1 / bin_size)
		bar_x = [k*bin_size for k in range(-n, n+1)]
		bar_y = [c[x] for x in bar_x]
		bar_x = [max(-1, min(1, x)) for x in bar_x]
		# bar_x, bar_y = [x[0] for x in sorted(c.items())], [x[1] for x in sorted(c.items())]
		bar_y = list(map(lambda x: x / sum(bar_y), bar_y))

		# plt.bar(bar_x, bar_y, width=bin_size, align="center")


		if args.data_types[i] == "original":
			label = args.data_name
		else:
			if args.data_types[i] == "NIMF_9999999999":
				label = "inf"
			else:
				label = args.data_types[i].split("_")[-1]
				label = label[:-6] + "." + label[-6] + "M"
			label = "NIMF($d_{max}=$" + label + ")"

		plt.plot(bar_x, bar_y, marker="o", label=label, markersize=0, linewidth=0.8)
	plt.title(f"{args.data_name} {args.cell_line} {regionType}")
	plt.xlabel("positive-negative gap ratio")
	plt.ylabel("freq rate")
	plt.xlim([-1.1, 1.1])
	plt.ylim([-0.05, 1.1])
	plt.grid(True)
	plt.legend()
	plt.savefig(output_path)




def make_PosNeg_figure(args):

	dfs = get_df(args)

	columnName = args.regionType + "_name"
	positive_negative_gap_ratio_by_df = [[] for i in range(len(dfs))]

	for i, df in enumerate(dfs):
		for regionName, subdf in df.groupby(columnName):
			posCnt = len(subdf[subdf["label"] == 1])
			negCnt = len(subdf[subdf["label"] == 0])

			positive_negative_gap_ratio_by_df[i].append((posCnt - negCnt) / max(posCnt, negCnt))

	make_positive_negative_gap_ratio_fig(args, positive_negative_gap_ratio_by_df, 0.05, regionType)


def make_distance_distribution_fugure_each_PN(data_name="BENGI", data_type="original", cell_line="GM12878", bin_size=1000, bottom_p = 100, **_):
	filename = os.path.join(os.path.dirname(__file__), data_name, data_type, f"{cell_line}.csv")
	df = pd.read_csv(filename, usecols=["label", "enhancer_distance_to_promoter"])
	pos_dist = df[df["label"]==1]["enhancer_distance_to_promoter"].to_list()
	neg_dist = df[df["label"]==0]["enhancer_distance_to_promoter"].to_list()

	pos_dist = sorted(pos_dist)[:int(len(pos_dist) * (bottom_p / 100))]
	neg_dist = sorted(neg_dist)[:int(len(neg_dist) * (bottom_p / 100))]

	# print(pos_dist)

	pos_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), pos_dist))
	neg_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), neg_dist))

	# print(pos_dist)

	pos_X_max = max(pos_dist)
	neg_X_max = max(neg_dist)

	pos_c = collections.Counter(pos_dist)
	neg_c = collections.Counter(neg_dist)

	pos_X = []
	pos_Y = []
	neg_X = []
	neg_Y = []
	for n_bin in range(int(pos_X_max*10)//(bin_size*10)+1):
		x = n_bin * bin_size + (bin_size / 2)
		pos_X.append(x)
		pos_Y.append(pos_c[x])
		# print(pos_X, pos_Y)
	for n_bin in range(int(neg_X_max*10)//(bin_size*10)+1):
		x = n_bin * bin_size + (bin_size / 2)
		neg_X.append(x)
		neg_Y.append(neg_c[x])

	outdir = os.path.join(os.path.dirname(__file__), "figure", "distance_distribution_each_PN")
	os.makedirs(outdir, exist_ok=True)

	plt.figure()
	plt.bar(pos_X, pos_Y, width=bin_size, align="center", color="red")
	plt.xlabel(f"EP distance (bin size {bin_size})")
	plt.ylabel(f"freq")
	plt.title(f"{data_name} {cell_line}")
	plt.savefig(os.path.join(outdir, f"{data_name}_{cell_line}_{bin_size}_bottom{bottom_p}.png"))

	plt.figure()
	plt.bar(neg_X, neg_Y, width=bin_size, align="center", color="blue")
	plt.xlabel(f"EP distance (bin size {bin_size})")
	plt.ylabel(f"freq")
	plt.title(f"{data_name} {data_type} {cell_line}")
	plt.savefig(os.path.join(outdir, f"N_{data_name}_{data_type}_{cell_line}_bin{bin_size}_bottom{bottom_p}.png"))


def make_distance_distribution_fugure_together_PN(data_name="BENGI", data_type="original", cell_line="GM12878", bin_size=1000, bottom_p = 100, **_):
	filename = os.path.join(os.path.dirname(__file__), data_name, data_type, f"{cell_line}.csv")
	df = pd.read_csv(filename, usecols=["label", "enhancer_distance_to_promoter"])

	df = df.sort_values(by="enhancer_distance_to_promoter").reset_index()
	df = df.loc[:int(len(df) * (bottom_p / 100)), :]

	
	pos_dist = df[df["label"]==1]["enhancer_distance_to_promoter"].to_list()
	neg_dist = df[df["label"]==0]["enhancer_distance_to_promoter"].to_list()

	print(f"positive pair: {len(pos_dist)}")
	print(f"negative pair: {len(neg_dist)}")

	pos_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), pos_dist))
	neg_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), neg_dist))

	# print(pos_dist)

	pos_X_max = max(pos_dist)
	neg_X_max = max(neg_dist)

	pos_c = collections.Counter(pos_dist)
	neg_c = collections.Counter(neg_dist)

	pos_X = []
	pos_Y = []
	neg_X = []
	neg_Y = []
	for n_bin in range(int(pos_X_max*10)//(bin_size*10)+1):
		x = n_bin * bin_size + (bin_size / 2)
		pos_X.append(x)
		pos_Y.append(pos_c[x])
		# print(pos_X, pos_Y)
	for n_bin in range(int(neg_X_max*10)//(bin_size*10)+1):
		x = n_bin * bin_size + (bin_size / 2)
		neg_X.append(x)
		neg_Y.append(neg_c[x])

	outdir = os.path.join(os.path.dirname(__file__), "figure", "distance_distribution_together_PN")
	os.makedirs(outdir, exist_ok=True)

	plt.figure()
	plt.bar(pos_X, pos_Y, width=bin_size, align="center", color="red", alpha=0.6, label="positive")
	plt.bar(neg_X, neg_Y, width=bin_size, align="center", color="blue", alpha=0.6, label="negative")
	plt.xlabel(f"EP distance (bin size {bin_size})")
	plt.ylabel(f"freq")
	plt.title(f"{data_name} {data_type} {cell_line}")
	plt.legend()
	plt.savefig(os.path.join(outdir, f"{data_name}_{data_type}_{cell_line}_bin{bin_size}_bottom{bottom_p}.png"))





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--data_name", help="", default="TargetFinder")
	parser.add_argument("--data_types", help="")
	parser.add_argument("--data_type", help="")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--regionType", default="enhancer")
	args = parser.parse_args()

	cell_line_list = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]

	# for name in ["BENGI", "TargetFinder"]:
	# 	for regionType in ["enhancer", "promoter"]:
	# 		args.data_name = name
	# 		args.regionType = regionType
	# 		args.data_types = ["original", "NIMF_2500000", "NIMF_5000000", "NIMF_10000000", "NIMF_9999999999"]
	# 		print(f"{args.data_name} {args.regionType}")
	# 		make_PosNeg_figure(args)

	for name in ["BENGI", "TargetFinder"]:
		for type in ["original", "NIMF_9999999999", "cmn_test_pair"]:
			for cell in cell_line_list:
				args.data_name = name
				args.data_type = type
				args.cell_line = cell
				# print(**vars(args))
				make_distance_distribution_fugure_each_PN(**vars(args), bottom_p=100, bin_size=1000)
				make_distance_distribution_fugure_together_PN(**vars(args), bottom_p=100, bin_size=10000)
	exit()


	for data_name in ["BENGI", "TargetFinder"]:
		for  data_type in ["original", "NIMF_2500000", "NIMF_5000000", "NIMF_10000000", "NIMF_9999999999"]:
			for cell_line in ["GM12878"]:
				args.data_name = data_name
				args.data_type = data_type
				args.cell_line = cell_line
				make_PosNeg_matrix(args)


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