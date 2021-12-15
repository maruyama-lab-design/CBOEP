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
from utils import pickle_dump

import data_download


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

			
def make_training_txt_unused(args, cell_line):
	# 分類器学習の際の前処理
	# textfileにてペア情報を書き込む
	# ___training.txt_________________
	#	ENHANCER_521	PROMOTER_61		1
	# 	ENHANCER_1334	PROMOTER_129	1
	# 	ENHANCER_1335	PROMOTER_129	1
	#	:				:				:
	# ________________________________

	
	# 前工程で作ったcsvを読み込む
	enhancer_bed_table = pd.read_csv(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed.csv", usecols=["name_origin"])
	promoter_bed_table = pd.read_csv(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed.csv", usecols=["name_origin"])

	# トレーニングデータダウンロード
	data_download.download_training_data(args, cell_line)
	train_csv = pd.read_csv(f"{args.my_data_folder_path}/train/{cell_line}_train.csv")

	# ペア情報を training.txt にメモ
	fout = open('training.txt','w')
	for _, row_data in train_csv.iterrows():
		enhancer_index_list = enhancer_bed_table[enhancer_bed_table["name_origin"] == row_data["enhancer_name"]].index.tolist()
		if len(enhancer_index_list) > 1: # ありえないが一応
			print("エラー!!")
			exit()
		elif len(enhancer_index_list) == 0: # たまにトレーニングデータに書かれている領域がない場合がある
			print(row_data["enhancer_name"])
			continue
		enhancer_index = enhancer_index_list[0]
		enhancer_tag = "enhancer_" + str(enhancer_index)

		promoter_index_list = promoter_bed_table[promoter_bed_table["name_origin"] == row_data["promoter_name"]].index.tolist()
		if len(promoter_index_list) > 1:
			print("エラー!!")
			exit()
		elif len(promoter_index_list) == 0:
			print(row_data["promoter_name"])
			continue
		promoter_index = promoter_index_list[0]
		promoter_tag = "promoter_" + str(promoter_index)
		label = row_data["label"]

		# enhancer の ~ 番目と promoter の ~ 番目 は pair/non-pair であるというメモを書き込む
		fout.write(enhancer_tag +'\t'+ promoter_tag + '\t' + str(label) + '\n')
	fout.close()



def make_training_txt_unused2(args, cell_line):
	# 分類器学習の際の前処理
	# textfileにてペア情報を書き込む
	# ___training.txt_________________
	#	ENHANCER_521	PROMOTER_61		1
	# 	ENHANCER_1334	PROMOTER_129	1
	# 	ENHANCER_1335	PROMOTER_129	1
	#	:				:				:
	# ________________________________

	global positive_num,negative_num
	print("トレーニングデータを参照してtxtfileを作成します.")
	positive_num = 0
	negative_num = 0
	
	# 前工程で作ったcsvを読み込む
	enhancer_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.csv", usecols=["name", "tag", "n_cnt"])
	promoter_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv", usecols=["name", "tag", "n_cnt"])

	if not os.path.isfile(f"{args.my_data_folder_path}/train/{cell_line}_train.csv"):
		print("トレーニングデータが見つかりません. ダウンロードします.") # ep2vecよりダウンロード
		os.system(f"wget https://raw.githubusercontent.com/wanwenzeng/ep2vec/master/{cell_line}train.csv -O {args.my_data_folder_path}/train/{cell_line}_train.csv")
	train_csv = pd.read_csv(f"{args.my_data_folder_path}/train/{cell_line}_train.csv", usecols=["bin", "enhancer_name", "promoter_name", "label"])

	enhancer_names = enhancer_table["name"].to_list()
	enhancer_tags = enhancer_table["tag"].to_list()
	promoter_names = promoter_table["name"].to_list()
	promoter_tags = promoter_table["tag"].to_list()
	enhancer_n_cnts = enhancer_table["n_cnt"].to_list()
	promoter_n_cnts = promoter_table["n_cnt"].to_list()

	# ペア情報を training.txt にメモ
	fout = open('training.txt','w')
	for _, data in train_csv.iterrows(): # train.csv を1行ずつ読み込み

		#学習に使うペアの領域情報
		train_enhancer_name = data["enhancer_name"]
		train_promoter_name = data["promoter_name"]

		enhancer_tag = "nan" # 初期化
		promoter_tag = "nan" # 初期化
		if train_enhancer_name in enhancer_names:
			index = enhancer_names.index(train_enhancer_name) # トレーニングデータのenhancer名から何番目のenhancerであるかを調べる
			if enhancer_n_cnts[index] == 0: # nを含むものはdoc2vecの学習に使ってないのでnを含んでないもののみを使う
				enhancer_tag = enhancer_tags[index]

		if train_promoter_name in promoter_names:
			index = promoter_names.index(train_promoter_name)  # トレーニングデータのpromoter名から何番目のpromoterであるかを調べる
			if promoter_n_cnts[index] == 0: # nを含むものはdoc2vecの学習に使ってないのでnを含んでないもののみを使う
				promoter_tag = promoter_tags[index]

		if enhancer_tag == "nan" or promoter_tag == "nan":
			continue

		label = data["label"]
		# enhancer の ~ 番目と promoter の ~ 番目 は pair/non-pair であるというメモを書き込む
		fout.write(str(enhancer_tag)+'\t'+str(promoter_tag)+'\t'+str(label)+'\n')

		if label == 1: # 正例
			positive_num += 1
		else: # 負例
			negative_num += 1

	print(f"正例数: {positive_num}")
	print(f"負例数: {negative_num}")


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
			y_score = classifier.decision_function(x_test) # 閾値なし
			y_pred = classifier.predict(x_test) # 閾値 0.5 predict

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
	make_training_df(args, cell_line) # train_csvをダウンロード&修正

	train_path = os.path.join(args.my_data_folder_path, "train", f"{cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_chrom", "enhancer_tag", "promoter_tag", "label"]) # train_csvを読み込み

	# paragraph vector モデルのロード
	d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{args.output}.d2v")
	d2v_model = Doc2Vec.load(d2v_model_path)

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

	# 分類器を用意 # TO DO optionで分けられるように
	classifier = my_classifier(args)
	# estimator = KNeighborsClassifier(n_neighbors=5) # k近傍法

	# cross validation training
	print("training classifier...")
	result_dicts = my_cross_validation(args, classifier, X_df, Y_df) # ここで学習開始

	# save result_dicts to ~.pickle
	save_filename = os.path.join(args.my_data_folder_path, "result", f"{args.output}.pickle")
	print(f"saving result to {save_filename}...")
	pickle_dump(result_dicts, save_filename)
	print("saved!!")

