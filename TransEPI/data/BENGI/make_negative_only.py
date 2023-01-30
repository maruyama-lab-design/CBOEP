import pandas as pd
import argparse
import os
import pulp

import glob




def extract_negative_pairs(args):
	# data directory を この~.pyと同じ場所に作成
	output_dir = os.path.join(os.path.dirname(__file__), "original", "negative_only")
	os.system(f"mkdir {output_dir}")
	# 保存先
	output_path = os.path.join(output_dir, args.filename)
	# if os.path.exists(output_path):
	# 	return

	data_path = os.path.join(os.path.dirname(__file__), "original", args.filename)
	df = pd.read_table(data_path, header=None, names=["label", "distance", "enh_chrom", "enh_start", "enh_end", "enh_name", "prm_chrom", "prm_start", "prm_end", "prm_name"])

	# 負例のみを取り出す
	negativeOnly_df = df[df["label"] == 0]
	negativeOnly_df.to_csv(output_path, index=False, header=False, sep="\t")



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--filename", default="")
	args = parser.parse_args()

	files = glob.glob(os.path.join(os.path.dirname(__file__), "original", "*.tsv"))
	for file in files:
		args.filename = os.path.basename(file)
		print(f"negative only dataset from {args.filename}...")
		extract_negative_pairs(args)