# sequence name をキーとしてベクトルを返す directory を作成し、保存

import pickle
import build_model as bm
import numpy as np
import pandas as pd
import argparse
import os
import glob


def concat_data(args):
	# 学習データが複数ファイルの場合、連結してdataframeで返す
	# print(os.path.join(os.path.dirname(__file__), "tarining_data", args.dataname, args.datatype, f"{args.cell_line}*.csv"))
	data_list = glob.glob(os.path.join(os.path.dirname(__file__), "training_data", args.dataname, args.datatype, f"{args.cell_line}*.csv"))
	df_list = []
	row_size = 0
	for data in data_list:
		df_tmp = pd.read_csv(data, sep=",", usecols=["enhancer_chrom","enhancer_start","enhancer_end","enhancer_name","label","promoter_chrom","promoter_start","promoter_end","promoter_name"])
		row_size += len(df_tmp)
		df_list.append(df_tmp)
	df = pd.concat(df_list)
	assert len(df) == row_size
	# print(df.head())

	print(f"pair size: {len(df)}")
	return df


def seq2vector(seq, length):
	dic = {
		"a":np.array([1, 0, 0, 0]),
		"c":np.array([0, 1, 0, 0]),
		"g":np.array([0, 0, 1, 0]),
		"t":np.array([0, 0, 0, 1]),
		"n":np.array([0, 0, 0, 0])
	}

	vector = np.zeros((length, 4))
	for i in range(length):
		acgt = ""
		if i < len(seq):
			acgt = seq[i]
		else:
			acgt = "n"
		vector[i] = dic[acgt]

	return vector


def get_name2seq(enh_list, prm_list):
	# return dict
	# which return sequence(acgt) from sequence_name

	# enh_list => [
	#   {エンハンサー名: , 染色体番号: , 開始位置: , 終了位置: }
	# ]

	name2seq_enh, name2seq_prm = {}, {}
	key_cash = ""
	
	chr2seq = None
	with open(os.path.join(os.path.dirname(__file__), "raw_seq", "chrom2seq.pkl"), "rb") as tf:
		chr2seq = pickle.load(tf)

	for enh in enh_list:
		name = enh["name"]
		chrom = enh["chrom"]
		start = enh["start"]
		end = enh["end"]
		seq = chr2seq[chrom][start:end]

		name2seq_enh[name] = seq

	for prm in prm_list:
		name = prm["name"]
		chrom = prm["chrom"]
		start = prm["start"]
		end = prm["end"]
		seq = chr2seq[chrom][start:end]
		
		name2seq_prm[name] = seq

	return name2seq_enh, name2seq_prm


def get_name2vec(enh_list, prm_list):
	# return dict
	# which return vector from sequence_name

	name2vec = {}
	name2seq_enh, name2seq_prm = get_name2seq(enh_list, prm_list)
	for name, seq in name2seq_enh.items():
		name2vec[name] = seq2vector(seq, 3000)
	for name, seq in name2seq_prm.items():
		name2vec[name] = seq2vector(seq, 2000)
	
	return name2vec


def make_chrom2dataset(args):
	# return dict
	# which return dataset from chromosome
	# dataset has keys such as "X_enhancers", "X_promoters", and "label"

	chr2dataset = {
		"chr1": {
			"X_enhancers":[],
			"X_promoters":[],
			"labels":[]
		},
		"chr2": {
			"X_enhancers":[],
			"X_promoters":[],
			"labels":[]
		},
		# 続く．．．
	}

	# 重要！！ここでデータ読み込み
	df = concat_data(args)
	enh_list = []
	prm_list = []

	for _, data in df.iterrows():

		enh_info_dic = {
			"name":data["enhancer_name"],
			"chrom":data["enhancer_chrom"],
			"start":data["enhancer_start"],
			"end":data["enhancer_end"],
		}

		prm_info_dic = {
			"name":data["promoter_name"],
			"chrom":data["promoter_chrom"],
			"start":data["promoter_start"],
			"end":data["promoter_end"],
		}

		enh_list.append(enh_info_dic)
		prm_list.append(prm_info_dic)


	name2vec = get_name2vec(enh_list, prm_list)


	for _, data in df.iterrows():
		enh_name = data["enhancer_name"]
		prm_name = data["promoter_name"]

		enh_vector = name2vec[enh_name]
		prm_vector = name2vec[prm_name]
		label = data["label"]
		chromosome = data["enhancer_chrom"]

		if chromosome not in chr2dataset:
			chr2dataset[chromosome] = {
				"X_enhancers":[],
				"X_promoters":[],
				"labels":[]
			}

		chr2dataset[chromosome]["X_enhancers"].append(enh_vector)
		chr2dataset[chromosome]["X_promoters"].append(prm_vector)
		chr2dataset[chromosome]["labels"].append(label)

	with open(os.path.join(args.output_dir, f"{args.cell_line}_chrom2dataset.pkl"), "wb") as tf:
		pickle.dump(chr2dataset, tf)
	# return chr2dataset







if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataname", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--datatype", help="どのデータセットを使うか", default="original")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--output_dir", default="")
	args = parser.parse_args()

	# for cell_line in ["GM12878", "K562", "HeLa", "HUVEC", "IMR90", "NHEK"]:
	for cell_line in ["GM12878"]:
		for dataset in ["TargetFinder", "BENGI"]:
			for datatype in ["original", "maxflow", "constrained(all_regions_have_pos_pair)", "EP2vec"]:

				args.cell_line = cell_line
				args.dataname = dataset
				args.datatype = datatype
				args.output_dir = os.path.join(os.path.dirname(__file__), "chrom2dataset", args.dataname, args.datatype)


				os.makedirs(args.output_dir, exist_ok=True)

				make_chrom2dataset(args)
