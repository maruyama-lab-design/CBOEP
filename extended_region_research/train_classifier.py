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

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from t_sne import t_SNE

import data_download


def make_training_df(args, cell_line):
	# 分類器学習の際の前処理
	
	print("トレーニングデータをcsvファイルにて書き込み開始")
	
	# トレーニングデータをtargetfinderからダウンロード & 読み込み
	data_download.download_training_data(args, cell_line)
	train_path = os.path.join(args.my_data_folder_path, "train", f"{cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_name", "promoter_name", "label"])

	train_df["enhancer_tag"] = -1 # カラムの追加
	train_df["promoter_tag"] = -1 # カラムの追加

	for region_type in ["enhancer", "promoter"]:

		region_bed_path = os.path.join(args.my_data_folder_path, "bed", region_type, f"{cell_line}_{region_type}s.bed.csv")
		region_bed_df = pd.read_csv(region_bed_path, usecols=["name_origin"])

		for region_index, row_data in region_bed_df.iterrows():
			if region_type == "enhancer":
				train_index_list = train_df[train_df["enhancer_name"] == row_data["name_origin"]].index.tolist()
				if len(train_index_list) == 0:
					continue
				train_df["enhancer_tag"][train_index_list] = "enhancer_" + str(region_index)
			elif region_type == "promoter":
				train_index_list = train_df[train_df["promoter_name"] == row_data["name_origin"]].index.tolist()
				if len(train_index_list) == 0:
					continue
				train_df["promoter_tag"][train_index_list] = "promoter_" + str(region_index)

	train_df.to_csv(train_path)
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


def train(args, cell_line):
	make_training_df(args, cell_line)

	X = np.empty((0, args.embedding_vector_dimention * 2))
	Y = np.empty(0)

	if args.share_doc2vec: #エンハンサーとプロモーター共存
		# paragraph vector モデルのロード
		d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
		d2v_model = Doc2Vec.load(d2v_model_path)
		
		train_path = os.path.join(args.my_data_folder_path, "train", f"{cell_line}_train.csv")
		train_df = pd.read_csv(train_path, usecols=["enhancer_tag", "promoter_tag", "label"])
		
		for pair_index, row_data in train_df.iterrows():

			enhancer_tag = str(row_data["enhancer_tag"]) # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = str(row_data["promoter_tag"]) # "PROMOTER_0" などのembedding vector タグ
			label = int(row_data["label"])

			if enhancer_tag == "-1" or promoter_tag == "-1":
				print(f"{pair_index}番目のペア スキップ")
				continue

			# if (enhancer_tag not in paragraph_tag_list) or (promoter_tag not in paragraph_tag_list):
			# 	continue

			enhancer_vec = d2v_model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = d2v_model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			concat_vec = np.column_stack((enhancer_vec,promoter_vec))
			X = np.append(X, concat_vec, axis=0)
			Y = np.append(Y, label) # 正例か負例か
	else: # エンハンサーとプロモーター別々 # 消してもいいかな？
		# paragraph vector モデルのロード
		enhancer_model = Doc2Vec.load(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
		promoter_model = Doc2Vec.load(f"{args.my_data_folder_path}/d2v/{cell_line},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")

		enhancer_paragraph_tag_list = list(enhancer_model.dv.doctags)
		promoter_paragraph_tag_list = list(promoter_model.dv.doctags)

		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		for i, line in enumerate(fin):
			data = line.strip().split()
			enhancer_tag = data[0] # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = data[1] # "PROMOTER_0" などのembedding vector タグ
			label = int(data[2])

			# if (enhancer_tag not in enhancer_paragraph_tag_list) or (promoter_tag not in promoter_paragraph_tag_list):
			# 	continue

			enhancer_vec = enhancer_model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = promoter_model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			concat_vec = np.column_stack((enhancer_vec,promoter_vec))
			X = np.append(X, concat_vec, axis=0)
			Y = np.append(Y, label) # 正例か負例か

	

	# 分類器を用意 # TO DO optionで分けられるように
	estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	# estimator = KNeighborsClassifier(n_neighbors=5) # k近傍法

	# t_sneにて図示
	t_SNE(args, X, Y)

	# 評価する指標
	score_funcs = ['f1', 'roc_auc', 'average_precision']
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	print("training classifier...")
	scores = cross_validate(estimator, X, Y, scoring = score_funcs, cv = cv, n_jobs = -1) # ここで学習開始

	# 得られた指標を出力する & 結果の記録
	print('F1:', scores['test_f1'].mean())
	print('auROC:', scores['test_roc_auc'].mean())
	print('auPRC:', scores['test_average_precision'].mean())
	f1 = scores['test_f1']
	f1 = np.append(f1, scores['test_f1'].mean())
	auROC = scores['test_roc_auc']
	auROC = np.append(auROC, scores['test_roc_auc'].mean())
	auPRC =  scores['test_average_precision']
	auPRC = np.append(auPRC, scores['test_average_precision'].mean())
	result = pd.DataFrame(
		{
		"F1": f1,
		"auROC": auROC,
		"auPRC": auPRC,
		},
		index = ["1-fold", "2-fold", "3-fold", "4-fold", "5-fold", "6-fold", "7-fold", "8-fold", "9-fold", "10-fold", "mean"]	
	)

	result_path = os.path.join(args.my_data_folder_path, "result", f"{args.output}.csv")
	result.to_csv(result_path) # 結果をcsvで保存

def train_unused(args, cell_line):
	make_training_txt(args, cell_line)
	print("training classifier...")

	X = np.empty((0, args.embedding_vector_dimention * 2))
	Y = np.empty(0)

	if args.share_doc2vec: #エンハンサーとプロモーター共存
		# paragraph vector モデルのロード
		# os path join を使う
		model = Doc2Vec.load(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
		# paragraph_tag_list = list(model.dv.index_to_key)
		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		for _, line in enumerate(fin):
			data = line.strip().split()
			enhancer_tag = data[0] # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = data[1] # "PROMOTER_0" などのembedding vector タグ
			label = int(data[2])

			# if (enhancer_tag not in paragraph_tag_list) or (promoter_tag not in paragraph_tag_list):
			# 	continue

			enhancer_vec = model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			concat_vec = np.column_stack((enhancer_vec,promoter_vec)) # 要チェック
			X = np.append(X, concat_vec, axis=0)
			Y = np.append(Y, label) # 正例か負例か
	else: # エンハンサーとプロモーター別々
		# paragraph vector モデルのロード
		enhancer_model = Doc2Vec.load(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
		promoter_model = Doc2Vec.load(f"{args.my_data_folder_path}/d2v/{cell_line},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")

		enhancer_paragraph_tag_list = list(enhancer_model.dv.doctags)
		promoter_paragraph_tag_list = list(promoter_model.dv.doctags)

		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		for i, line in enumerate(fin):
			data = line.strip().split()
			enhancer_tag = data[0] # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = data[1] # "PROMOTER_0" などのembedding vector タグ
			label = int(data[2])

			# if (enhancer_tag not in enhancer_paragraph_tag_list) or (promoter_tag not in promoter_paragraph_tag_list):
			# 	continue

			enhancer_vec = enhancer_model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = promoter_model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			concat_vec = np.column_stack((enhancer_vec,promoter_vec))
			X = np.append(X, concat_vec, axis=0)
			Y = np.append(Y, label) # 正例か負例か

	

	# 分類器を用意
	estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	# estimator = KNeighborsClassifier(n_neighbors=5) # k近傍法

	# t_sneにて図示
	t_SNE(args, X, Y)

	# 評価する指標
	score_funcs = ['f1', 'roc_auc', 'average_precision']
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	print("training classifier...")
	scores = cross_validate(estimator, X, Y, scoring = score_funcs, cv = cv, n_jobs = -1) # ここで学習開始

	# 得られた指標を出力する & 結果の記録
	print('F1:', scores['test_f1'].mean())
	print('auROC:', scores['test_roc_auc'].mean())
	print('auPRC:', scores['test_average_precision'].mean())
	f1 = scores['test_f1']
	f1 = np.append(f1, scores['test_f1'].mean())
	auROC = scores['test_roc_auc']
	auROC = np.append(auROC, scores['test_roc_auc'].mean())
	auPRC =  scores['test_average_precision']
	auPRC = np.append(auPRC, scores['test_average_precision'].mean())
	result = pd.DataFrame(
		{
		"F1": f1,
		"auROC": auROC,
		"auPRC": auPRC,
		},
		index = ["1-fold", "2-fold", "3-fold", "4-fold", "5-fold", "6-fold", "7-fold", "8-fold", "9-fold", "10-fold", "mean"]	
	)

	result.to_csv(f"{args.my_data_folder_path}/result/{args.output}.csv") # 結果をcsvで保存

# 入力データと中間で作られるファイルを全部おく