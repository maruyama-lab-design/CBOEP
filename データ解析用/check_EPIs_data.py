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

def make_heatMap(VV, x_Max, y_Max, output_path):
	axis_Max = max(x_Max, y_Max)
	fontsize = set_fontsize(axis_Max)

	data = VV[:axis_Max+1, :axis_Max+1] # 二次元配列です

	mask = np.zeros_like(data)
	mask[np.where(data==0)] = True    

	plt.figure(figsize=(21, 14))
	fig, ax = plt.subplots()
	sns.heatmap(data, annot=True, square=True, annot_kws={"fontsize":fontsize}, fmt="d", cmap="Blues", linewidths=.1, linecolor='black', mask=mask, cbar = False)
	ax.invert_yaxis()
	# ax.set_title(f"{args.cell_line} pos-neg cnt by each {regionType}")
	ax.set_xlabel("positive example count")
	ax.set_ylabel("negative example count")
	plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')

	
def make_PosNeg_figure(df, outdir):


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

				PosNeg_cnt_by_chrom[negCnt][posCnt] += 1

				posCnt_Max_by_chrom = max(posCnt_Max_by_chrom, posCnt)
				negCnt_Max_by_chrom = max(negCnt_Max_by_chrom, negCnt)

			outfile = os.path.join(outdir, f"{chrom}_{regionType}.png")
			make_heatMap(PosNeg_cnt_by_chrom, posCnt_Max_by_chrom, negCnt_Max_by_chrom, outfile)
			
			PosNeg_cnt += PosNeg_cnt_by_chrom
			posCnt_Max = max(posCnt_Max, posCnt_Max_by_chrom)
			negCnt_Max = max(negCnt_Max, negCnt_Max_by_chrom)

		outfile = os.path.join(outdir, f"chrAll_{regionType}.png")
		make_heatMap(PosNeg_cnt, posCnt_Max, negCnt_Max, outfile)


def make_biasError_table(dataname, datatype_list, cell_line, outdir):
	table_dir = {
		"dataset": [],
		"datatype": [],
		"distance limit": [],
		"positive": [],
		"negative": [],
		"enhancer bias error average": [],
		"enhancer bias error standard deviation": [],
		"promoter bias error average": [],
		"promoter bias error standard deviation": []
	}

	for datatype in datatype_list:
		infiles = glob.glob(os.path.join(os.path.dirname(__file__), "data", dataname, datatype, f"{cell_line}*.csv"))
		df = files2df(infiles)

		table_dir["dataset"].append(dataname)
		if datatype[0] == "m":
			table_dir["datatype"].append(datatype.split("_")[0])
			table_dir["distance limit"].append(datatype.split("_")[1])
		else:
			table_dir["datatype"].append(datatype)
			table_dir["distance limit"].append("")

		table_dir["positive"].append(len(df[df["label"] == 1]))
		table_dir["negative"].append(len(df[df["label"] == 0]))

		for regionType in ["enhancer", "promoter"]:
			PosNeg_cnt = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
			columnName = regionType + "_name"

			pair_cnt = 0

			for chrom, subdf in df.groupby("enhancer_chrom"):
				PosNeg_cnt_by_chrom = np.zeros((1000, 1000), dtype="int64") # 大きめに用意
				for regionName, subsubdf in subdf.groupby(columnName):
					posCnt = len(subsubdf[subsubdf["label"] == 1])
					negCnt = len(subsubdf[subsubdf["label"] == 0])

					PosNeg_cnt_by_chrom[negCnt][posCnt] += 1
					pair_cnt += 1

				
				PosNeg_cnt += PosNeg_cnt_by_chrom
			
			error_list = np.empty(0)
			for i in range(1000):
				for j in range(1000):
					error = abs(i - j)
					for k in range(PosNeg_cnt[i][j]):
						error_list = np.append(error_list, error)

			assert pair_cnt == len(error_list)

			# table_dir["region"].append(regionType)
			table_dir[f"{regionType} bias error average"].append(error_list.mean())
			table_dir[f"{regionType} bias error standard deviation"].append(error_list.std())


	outfile = os.path.join(outdir, f"{dataname}-{cell_line}_bias_error.csv")
	os.makedirs(os.path.dirname(outfile), exist_ok=True)
	table = pd.DataFrame(table_dir)
	table.to_csv(outfile, index=False)
	with pd.ExcelWriter(os.path.join(outdir, f"bias_error.xlsx"), mode='a', if_sheet_exists="replace") as writer:
		table.to_excel(writer, sheet_name=f"{dataname}-{cell_line}")



def files2df(infiles):
	outdf = None
	for i, infile in enumerate(infiles):
		if i == 0:
			outdf = pd.read_csv(infile, usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
		else:
			tmpdf = pd.read_csv(infile, usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
			outdf = pd.concat([outdf, tmpdf], axis=0)

	return outdf



if __name__ == '__main__':
	
	dataname_list = ["BENGI", "TargetFinder"]
	datatype_list = ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", "maxflow_9999999999"]
	cell_line_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]
	# for dataname in dataname_list:
	# 	for datatype in datatype_list:
	# 		for cell_line in cell_line_list:
	# 			infiles = glob.glob(os.path.join(os.path.dirname(__file__), "data", dataname, datatype, f"{cell_line}*.csv"))
	# 			df = files2df(infiles)
	# 			# outdir = os.path.join(os.path.dirname(__file__), "fig", dataname, datatype, cell_line)
	# 			# os.makedirs(outdir, exist_ok=True)
	# 			# make_PosNeg_figure(df, outdir)
	# 			outdir = os.path.join(os.path.dirname(__file__), "csv", dataname, datatype, cell_line)
	# 			make_biasError_table(df, dataname, datatype, cell_line, outdir)

	for dataname in dataname_list:
		for cell_line in cell_line_list:
			outdir = os.path.join(os.path.dirname(__file__), "csv")
			make_biasError_table(dataname, datatype_list, cell_line, outdir)

