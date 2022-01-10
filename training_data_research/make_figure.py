import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import argparse


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
	sns.heatmap(data, annot=True, square=True, annot_kws={"fontsize":fontsize}, fmt="d", cmap="Blues", linewidths=.1, linecolor='black', mask=mask, cbar = False)
	ax.invert_yaxis()
	ax.set_title(f"{args.cell_line} pos-neg cnt by each {regionType}")
	ax.set_xlabel("positive-cnt")
	ax.set_ylabel("negative-cnt")
	plt.savefig(output_path, format="png", dpi=300)
	plt.close('all')

	
def make_PosNeg_figure(args):
	output_dir = os.path.join(os.path.dirname(__file__), "figure", args.research_name, args.cell_line)
	os.system(f"mkdir -p {output_dir}")

	data_path = os.path.join(os.path.dirname(__file__), "training_data", args.research_name, f"{args.cell_line}_train.csv")
	df = pd.read_csv(data_path)

	for regionType in ["enhancer", "promoter"]:
		posCnt_Max = 0
		negCnt_Max = 0
		PosNeg_cnt = np.zeros((30, 30), dtype="int64") # 大きめに用意
		columnName = regionType + "_name"

		for chrom, subdf in df.groupby("enhancer_chrom"):
			posCnt_Max_by_chrom = 0
			negCnt_Max_by_chrom = 0
			PosNeg_cnt_by_chrom = np.zeros((30, 30), dtype="int64") # 大きめに用意
			for regionName, subsubdf in subdf.groupby(columnName):
				posCnt = len(subsubdf[subsubdf["label"] == 1])
				negCnt = len(subsubdf[subsubdf["label"] == 0])

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
	parser.add_argument("--research_name", help="", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	args = parser.parse_args()

	cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
	research_list = ["TargetFinder", "new"]
	for researchName in research_list:
		for cell_line in cell_line_list:
			args.research_name = researchName
			args.cell_line = cell_line
			make_PosNeg_figure(args)