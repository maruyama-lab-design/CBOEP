from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# classifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from t_sne import t_SNE

import data_download
from utils import pickle_dump, pickle_load


def make_training_df(args, cell_line):
	# 分類器学習の際の前処理
	# bed.csv から 各paragraph tag をそのindex値より取得し，train.csvに新しいcolumnとして書き込む
	
	print("トレーニングデータをcsvファイルにて書き込み開始")
	
	# トレーニングデータをtargetfinderからダウンロード & 読み込み
	data_download.download_training_data(args, cell_line)
	train_path = os.path.join(args.my_data_folder_path, "train", f"{cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"]) # original

	train_df["enhancer_tag"] = 'nan' # カラムの追加
	train_df["promoter_tag"] = 'nan' # カラムの追加

	for region_type in ["enhancer", "promoter"]:

		region_bed_path = os.path.join(args.my_data_folder_path, "bed", region_type, f"{cell_line}_{region_type}s.bed.csv")
		region_bed_df = pd.read_csv(region_bed_path, usecols=["name_origin"])

		for region_index, row_data in region_bed_df.iterrows():
			region_name = row_data["name_origin"]
			if region_type == "enhancer":
				train_index_list = train_df.query('enhancer_name == @region_name').index.tolist()
				if len(train_index_list) == 0:
					continue
				train_df.loc[train_index_list, "enhancer_tag"] = "enhancer_" + str(region_index)
			elif region_type == "promoter":
				train_index_list = train_df.query('promoter_name == @region_name').index.tolist()
				if len(train_index_list) == 0:
					continue
				train_df.loc[train_index_list, "promoter_tag"] = "promoter_" + str(region_index)

	
	drop_index_list = train_df.query('enhancer_tag == "nan" or promoter_tag == "nan"').index.tolist()
	train_df = train_df.drop(drop_index_list, axis=0)



	train_df.to_csv(train_path, index=False)
	print("トレーニングデータをcsvファイルにて書き込み終了")


def my_classifier(args): # 分類器を返す
	if args.classifier == "GBRT":
		return GradientBoostingClassifier(n_estimators = args.gbrt_tree_cnt, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)


def my_cross_validation(args, classifier, X_df, Y_df):

	result_dicts = {}

	if args.way_of_cv == "random": # random 10-fold cross-validation
		print("random 10-fold cross-validation...")
		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
		x = X_df.iloc[:, 1:].values
		y = Y_df.iloc[:, 0].values
		for fold, (train_index, test_index) in enumerate(cv.split(x, y)):
			print(f"fold {fold+1}...")
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]

			classifier.fit(x_train, y_train) # 学習
			y_pred = classifier.predict(x_test) # predict

			print(f"fold {fold+1}\nConfusion Matrix:")
			print(confusion_matrix(y_test, y_pred))

			print(f"fold {fold+1}\nClassification Report")
			print(classification_report(y_test, y_pred))

			result_dicts[f"fold{fold+1}"] = classification_report(y_test, y_pred, output_dict=True)
	elif args.way_of_cv == "split": # chromosomal-split cross-validateion
		print("chromosomal-split cross-validateion...")
		chromosomes = ["chr"+str(i) for i in range(1, 23)]
		chromosomes.append("chrX")
		df = pd.concat([X_df, Y_df], axis=1)
		df_groupby_chrom = df.groupby("chrom")

		test_chroms = [["chr1", "chr2"], ["chr3", "chr4"], ["chr5", "chr6"], ["chr7", "chr8"], ["chr9", "chr10"], ["chr11", "chr12"], ["chr13", "chr14"], ["chr15", "chr16"], ["chr17", "chr18"], ["chr19", "chr20"], ["chr21", "chr22"]]
		for fold, test_chrom in enumerate(test_chroms):
			print(f"fold {fold+1}...")
			print(f"test chromosome : {test_chrom}")
			test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]
			test_index = df.query('chrom == @test_chrom1 or chrom == @test_chrom2').index.tolist()
			test_df = df.iloc[test_index, :]
			train_df = df.drop(index=test_index)
			x_train = train_df.iloc[:, 1:-1].values
			y_train = train_df.iloc[:, -1].values
			x_test = test_df.iloc[:, 1:-1].values
			y_test = test_df.iloc[:, -1].values
			classifier.fit(x_train, y_train) # 学習
			y_pred = classifier.predict(x_test) # predict

			print(f"fold {fold+1}\nConfusion Matrix:")
			print(confusion_matrix(y_test, y_pred))

			print(f"fold {fold+1}\nClassification Report")
			print(classification_report(y_test, y_pred))

			result_dicts[f"fold{fold+1}"] = classification_report(y_test, y_pred, output_dict=True)

	return result_dicts



def train(args, cell_line):
	# make_training_df(args, cell_line) # train_csvをダウンロード&修正

	train_path = os.path.join(args.my_data_folder_path, "train", f"{cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_chrom", "enhancer_tag", "promoter_tag", "label"]) # train_csvを読み込み

	# paragraph vector モデルのロード
	d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{args.output}.d2v")
	d2v_model = Doc2Vec.load("data/d2v/K562,el=0,er=0,pl=0,pr=0,share_doc2vec=True,d=100kmer=normal,N=1.d2v")

	paragraph_tag_list = list(d2v_model.dv.index_to_key)

	# doc2vecに渡してないtrain data を削除
	drop_index_list = []
	for pair_index, row_data in train_df.iterrows():
		enhancer_tag = str(row_data["enhancer_tag"]) # "ENHANCER_0" などのembedding vector タグ
		promoter_tag = str(row_data["promoter_tag"]) # "PROMOTER_0" などのembedding vector タグ

		if (enhancer_tag not in paragraph_tag_list) or (promoter_tag not in paragraph_tag_list):
			drop_index_list.append(pair_index)
			
	train_df = train_df.drop(drop_index_list, axis=0)
	train_df = train_df.reset_index()

	print(f"ペア数: {len(train_df)}")

	X = np.zeros((len(train_df), args.embedding_vector_dimention * 2)) # 2d次元の埋め込みベクトルを入れるzero配列
	Y = np.zeros(len(train_df)) # labelを入れるzero配列
	chroms = np.empty(len(train_df), dtype=object) # 染色体番号を入れる空配列
	
	for pair_index, row_data in train_df.iterrows():

		enhancer_tag = str(row_data["enhancer_tag"]) # "ENHANCER_0" などのembedding vector タグ
		promoter_tag = str(row_data["promoter_tag"]) # "PROMOTER_0" などのembedding vector タグ
		label = int(row_data["label"])
		chrom = row_data["enhancer_chrom"]

		enhancer_vec = d2v_model.dv[enhancer_tag] # エンハンサーのembedding vector
		promoter_vec = d2v_model.dv[promoter_tag] # プロモーターのembedding vector
		enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
		promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
		concat_vec = np.column_stack((enhancer_vec,promoter_vec)) # concat
		X[pair_index] = concat_vec
		Y[pair_index] = label # 正例か負例か
		chroms[pair_index] = chrom # 染色体番号

	# t_sneにて図示
	t_SNE(args, X, Y)

	X_df = pd.DataFrame(X)
	Y_df = pd.DataFrame({"Y" : Y})
	chrom_df = pd.DataFrame({"chrom" : chroms})

	X_df = pd.concat([chrom_df, X_df], axis=1)
	print(X_df.head())
	print(Y_df.head())

	# 分類器を用意 # TO DO optionで分けられるように
	classifier = my_classifier(args)
	# estimator = KNeighborsClassifier(n_neighbors=5) # k近傍法

	# cross validation training
	print("training classifier...")
	result_dicts = my_cross_validation(args, classifier, X_df, Y_df) # ここで学習開始

	# save result_dicts to .pickle
	save_filename = "test.pickle"
	print(f"saving result to {save_filename}...")
	pickle_dump(result_dicts, "test.pickle")
	print("saved!!")



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--genome_browser_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest")
	parser.add_argument("-my_data_folder_path", help="自分のデータフォルダパス", default="/Users/ylwrvr/卒論/Koga_code/data")
	parser.add_argument("--make_directory", action="store_true", help="実験に必要なディレクトリ構成を作る")
	parser.add_argument("--download_reference_genome", action="store_true", help="リファレンスゲノムを外部からダウンロードするか")
	parser.add_argument("--share_doc2vec", action="store_true", help="エンハンサーとプロモーターを一つのdoc2vecに共存させるか")
	parser.add_argument("--cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-el", "--E_extended_left_length", type=int, default=0, help="エンハンサーの上流をどれだけ伸ばすか")
	parser.add_argument("-er", "--E_extended_right_length", type=int, default=0, help="エンハンサーの下流をどれだけ伸ばすか")
	parser.add_argument("-pl", "--P_extended_left_length", type=int, default=0, help="プロモーターの上流をどれだけ伸ばすか")
	parser.add_argument("-pr", "--P_extended_right_length", type=int, default=0, help="プロモーターの下流をどれだけ伸ばすか")
	parser.add_argument("--embedding_vector_dimention", type=int, default=100, help="paragraph vector の次元")
	parser.add_argument('--way_of_kmer', type=str, choices=['normal', 'random'], default="normal", help='k-merの切り方 固定長かランダム長か')
	parser.add_argument("--k", type=int, default=6, help="固定長のk-merの場合のk")
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--sentence_cnt", type=int, default=5, help="ランダム長のk-merの場合,一本のseqから生成されるのsentence個数")
	parser.add_argument("--k_min", type=int, default=3, help="ランダム長のk-merの場合のk_min")
	parser.add_argument("--k_max", type=int, default=6, help="ランダム長のk-merの場合のk_max")
	parser.add_argument("--classifier", type=str, choices=["GBRT"], default="GBRT", help="分類器に何を使うか")
	parser.add_argument("--way_of_cv", type=str, choices=["random", "split"], default="random", help="ランダムcross-valか，染色体番号ごとか")
	parser.add_argument("--gbrt_tree_cnt", type=int, default=4000, help="GBRTの木の数")
	parser.add_argument("--stage1_start_time", type=str, help="doc2vec開始時間")
	parser.add_argument("--stage1_end_time", type=str, help="doc2vec終了時間")
	parser.add_argument("--stage2_start_time", type=str, help="分類期学習開始時間")
	parser.add_argument("--stage2_end_time", type=str, help="分類期学習終了時間")
	parser.add_argument("--output", type=str, help="output名")
	args = parser.parse_args()

	if args.way_of_kmer == "normal":
		args.sentence_cnt = 1
		args.k_min = -1
		args.k_max = -1
	elif args.way_of_kmer == "random":
		args.k = -1
		args.stride = -1

	
	for cell_line in args.cell_line_list:
		train(args, cell_line)
		### save the contents of args to a log file.
		
