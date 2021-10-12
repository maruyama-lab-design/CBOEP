from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


def make_training_txt(args, cell_line):
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

		# 自分で作ったcsvの領域情報には領域を伸ばした後locationが書かれているので戻す
		train_enhancer_name = data["enhancer_name"].split("|")[1]
		chr, enhancer_range = train_enhancer_name.split(":")[0], train_enhancer_name.split(":")[1]
		train_enhancer_start, train_enhancer_end = int(enhancer_range.split("-")[0]) - args.E_extended_left_length, int(enhancer_range.split("-")[1]) + args.E_extended_right_length
		train_enhancer_name = chr + ":" + str(train_enhancer_start) + "-" + str(train_enhancer_end)

		train_promoter_name = data["promoter_name"].split("|")[1]
		chr, promoter_range = train_promoter_name.split(":")[0], train_promoter_name.split(":")[1]
		train_promoter_start, train_promoter_end = int(promoter_range.split("-")[0]) - args.P_extended_left_length, int(promoter_range.split("-")[1]) + args.P_extended_right_length
		train_promoter_name = chr + ":" + str(train_promoter_start) + "-" + str(train_promoter_end)

		enhancer_tag = "nan" # 初期化
		promoter_tag = "nan" # 初期化
		if train_enhancer_name in enhancer_names:
			index = enhancer_names.index(train_enhancer_name) # トレーニングデータのenhancer名から何番目のenhancerであるかを調べる
			if enhancer_n_cnts[index] > 0: # nを含むものを学習データからはずす
				continue
			enhancer_tag = enhancer_tags[index]

		if train_promoter_name in promoter_names:
			index = promoter_names.index(train_promoter_name)  # トレーニングデータのpromoter名から何番目のpromoterであるかを調べる
			if promoter_n_cnts[index] > 0: # nを含むものを学習データからはずす
				continue
			promoter_tag = promoter_tags[index]

		if enhancer_tag == "nan" or promoter_tag == "nan": 
			continue
		
		label = str(data["label"])

		# enhancer の ~ 番目と promoter の ~ 番目 は pair/non-pair であるというメモを書き込む
		fout.write(str(enhancer_tag)+'\t'+str(promoter_tag)+'\t'+label+'\n')

		if label == '1': # 正例
			positive_num = positive_num + 1
		else: # 負例
			negative_num = negative_num + 1

	print(f"正例数: {positive_num}")
	print(f"負例数: {negative_num}")


def train(args, cell_line):
	global positive_num, negative_num
	make_training_txt(args, cell_line)

	print("分類器を学習します.")

	arrays = np.zeros((positive_num+negative_num, args.embedding_vector_dimention*2)) # X (従属変数 後に EnhとPrmの embedding vector が入る)
	labels = np.zeros(positive_num+negative_num) # Y (目的変数 後に ペア情報{0 or 1}が入る)

	if args.share_doc2vec:
		# paragraph vector モデルのロード
		model = Doc2Vec.load(f'MyProject/data/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model')

		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		for i, line in enumerate(fin):
			data = line.strip().split()
			enhancer_tag = data[0] # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = data[1] # "PROMOTER_0" などのembedding vector タグ
			enhancer_vec = model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			arrays[i] = np.column_stack((enhancer_vec,promoter_vec)) # concat
			labels[i] = int(data[2]) # 正例か負例か
			i = i + 1
	else:
		# paragraph vector モデルのロード
		enhancer_model = Doc2Vec.load(f'MyProject/data/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}.model')
		promoter_model = Doc2Vec.load(f'MyProject/data/model/{cell_line}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model')

		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		for i, line in enumerate(fin):
			data = line.strip().split()
			enhancer_tag = data[0] # "ENHANCER_0" などのembedding vector タグ
			promoter_tag = data[1] # "PROMOTER_0" などのembedding vector タグ
			enhancer_vec = enhancer_model.dv[enhancer_tag] # エンハンサーのembedding vector
			promoter_vec = promoter_model.dv[promoter_tag] # プロモーターのembedding vector
			enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
			promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
			arrays[i] = np.column_stack((enhancer_vec,promoter_vec)) # concat
			labels[i] = int(data[2]) # 正例か負例か
			i = i + 1

	

	# 分類器を用意
	estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	# estimator = KNeighborsClassifier(n_neighbors=5)

	# 評価する指標
	score_funcs = ['f1', 'roc_auc', 'average_precision']
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	print("分類器学習中...")
	scores = cross_validate(estimator, arrays, labels, scoring = score_funcs, cv = cv, n_jobs = -1)

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
	result.to_csv(f"{args.my_data_folder_path}/result/{cell_line},el={str(args.E_extended_left_length)},er={str(args.E_extended_right_length)},pl={str(args.P_extended_left_length)},pr={str(args.P_extended_right_length)},share_doc2vec={str(args.share_doc2vec)}.csv")
