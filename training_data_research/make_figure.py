import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import argparse

def make_heatMap(VV, x_Max, y_Max, x_label, y_label, output_path):
	data = VV[:x_Max+2, :y_Max+2] # 二次元配列です

	mask = np.zeros_like(data)
	mask[np.where(data==0)] = True    

	plt.figure(figsize=(21, 14))
	sns.heatmap(data, annot=True, square=True, annot_kws={'size': 15}, fmt="d", cmap="spring", linewidths=1, linecolor='black', mask=mask)
	plt.savefig(output_path)
	# plt.show()
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

				PosNeg_cnt_by_chrom[posCnt][negCnt] += 1

				posCnt_Max_by_chrom = max(posCnt_Max_by_chrom, posCnt)
				negCnt_Max_by_chrom = max(negCnt_Max_by_chrom, negCnt)

			output_path = os.path.join(output_dir, f"{chrom}_{regionType}.png")
			make_heatMap(PosNeg_cnt_by_chrom, posCnt_Max_by_chrom, negCnt_Max_by_chrom, "pos", "neg", output_path)
			
			PosNeg_cnt += PosNeg_cnt_by_chrom
			posCnt_Max = max(posCnt_Max, posCnt_Max_by_chrom)
			negCnt_Max = max(negCnt_Max, negCnt_Max_by_chrom)

		output_path = os.path.join(output_dir, f"chrAll_{regionType}.png")
		make_heatMap(PosNeg_cnt, posCnt_Max, negCnt_Max, "pos", "neg", output_path)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--research_name", help="", default="new")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	args = parser.parse_args()

	cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
	for cell_line in cell_line_list:
		args.cell_line = cell_line
		make_PosNeg_figure(args)