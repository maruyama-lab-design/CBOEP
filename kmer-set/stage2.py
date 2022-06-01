# classifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from gensim.models import Doc2Vec

import os
import pandas as pd 
import numpy as np 
import argparse



def get_classifier(args):
	if args.classifier == "GBRT":
		return GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	elif args.classifier == "KNN":
		return KNeighborsClassifier(n_neighbors=args.knn_neighbor_cnt) # k近傍法
	elif args.classifier == "SVM":
		return svm.SVC(kernel='linear', random_state=None, probability=True)


def get_chr2dataset(args):

	print(f"{args.dataset} dataset から分類器用の辞書listを作成...")

	chr2dataset = {
		"chr1":{
			"X":[], # ２次元配列
			"Y":[] # １次元配列
		},
	}

	d2v = Doc2Vec.load(os.path.join(args.d2v_dir, args.cell_line, f"{args.k_stride_set}.d2v"))

	paragraphTag_list = list(d2v.dv.index_to_key)

	train_path = os.path.join(args.train_dir, args.dataset, f"{args.cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
	data_list = []


	for _, row_data in train_df.iterrows():
		chromosome = row_data["enhancer_chrom"]
		if chromosome not in chr2dataset:
			chr2dataset[chromosome] = {
				"X": [],
				"Y": []
			}


		enhName = row_data["enhancer_name"]
		prmName = row_data["promoter_name"]
		if (enhName not in paragraphTag_list) or (prmName not in paragraphTag_list):
			continue
		enhVec = list(d2v.dv[enhName])
		prmVec = list(d2v.dv[prmName])
		assert len(enhVec) == args.embedding_vector_dimention
		assert len(prmVec) == args.embedding_vector_dimention

		concatVec = enhVec + prmVec
		assert len(concatVec) == args.embedding_vector_dimention * 2
		chr2dataset[chromosome]["X"].append(concatVec)

		label = int(row_data["label"])
		chr2dataset[chromosome]["Y"].append(label)

	return chr2dataset


def get_weights(y):

	weights_dic = {
		0: 1 / (np.sum(y==0) / len(y)), # 負例重み
		1: 1 / (np.sum(y==1) / len(y)) # 正例重み
	}

	weights_arr = np.zeros(len(y))

	for i in range(len(y)):
		weights_arr[i] = weights_dic[y[i]]

	return weights_arr


def chromosomal_cv(args):
	chr2dataset = get_chr2dataset(args)
	test_chroms = [["chr1", "chr2"], ["chr3", "chr4"], ["chr5", "chr6"], ["chr7", "chr8"], ["chr9", "chr10"], ["chr11", "chr12"], ["chr13", "chr14"], ["chr15", "chr16"], ["chr17", "chr18"], ["chr19", "chr20"], ["chr21", "chr22"]]

	for fold, test_chrom in enumerate(test_chroms):
		output_path = os.path.join(args.result_dir, f"fold_{fold+1}.csv")
		print(f"fold {fold+1}...")
		print(f"test chromosome : {test_chrom}")
		test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]

		X_train, X_test = [], []
		Y_train, Y_test = [], []


		for chr_list in test_chroms:
			for chromosome in chr_list:
				if chromosome == test_chrom1 or chromosome == test_chrom2:
					X_test += chr2dataset[chromosome]["X"]
					Y_test += chr2dataset[chromosome]["Y"]
				else:
					X_train += chr2dataset[chromosome]["X"]
					Y_train += chr2dataset[chromosome]["Y"]

		
		X_train = np.array(X_train)
		X_test = np.array(X_test)
		Y_train = np.array(Y_train)
		Y_test = np.array(Y_test)

		assert len(X_train) == len(Y_train)
		assert len(X_test) == len(Y_test)

		print(f"train enhancers sample size: {len(X_train)}")

		classifier = get_classifier(args)
		weights = get_weights(Y_train)
		classifier.fit(X_train, Y_train, sample_weight=weights) # 学習

		print(f"5/{len(X_test)} of x_test")
		for i in range(5):
			print(X_test[i], f"label={Y_test[i]}")

		y_pred = classifier.predict_proba(X_test) # 予測
		y_pred = [prob[1] for prob in y_pred] # 正例確率のみを抽出
		# print(y_pred)

		result_df = pd.DataFrame(
			{
				"y_test": Y_test,
				"y_pred": y_pred
			},
			index=None
		)
		result_df.to_csv(output_path)
		print(f"saved in {output_path}")


def stage2(args):
	chromosomal_cv(args)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--ratio", default=1)
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k", help="k-merのk", type=int, default=6)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--k_list", help="k-merのk", default="1,2,3,4,5,6")
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--kmax", help="k-merのk", type=int, default=6)
	parser.add_argument("--kmin", help="k-merのk", type=int, default=3)
	parser.add_argument("--sentenceCnt", help="何個複製するか", type=int, default=3)
	parser.add_argument("--way_of_kmer", choices=["normal", "random"], default="normal")
	parser.add_argument("--vector_size", help="分散表現の次元", type=int, default=100)
	parser.add_argument("--way_of_cv", help="染色体毎かランダムか", choices=["chromosomal", "random"], default="chromosomal")
	parser.add_argument("--classifier", type=str, choices=["GBRT", "KNN", "SVM"], default="GBRT", help="分類器に何を使うか")
	parser.add_argument("--gbrt_tree_cnt", type=int, default=1000, help="GBRTの木の数")
	parser.add_argument("--knn_neighbor_cnt", type=int, default=5, help="k近傍法の近傍数")
	args = parser.parse_args()

	dataset_list = ["TargetFinder"]
	cell_line_list = ["IMR90", "HUVEC", "HeLa-S3"]
	cell_line_list = ["HUVEC", "HeLa-S3"]

	args.classifier = "GBRT"
	args.gbrt_tree_cnt = 4000
	args.way_of_cv = "chromosomal"
	# k_mer_set = ["1", "2", "3", "4", "5", "6", "1,2,3,4,5,6", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6"]
	k_mer_set = ["6"]
	# k_mer_set = ["1,2,3,4,5,6,7,8", "7", "8"]
	args.stride = 1
	for dataset in dataset_list:
		for cl in cell_line_list:
			for ratio in [1]:
				for k_list in k_mer_set:
					args.dataset = dataset
					args.cell_line = cl
					args.ratio = ratio
					args.k_list = k_list
					# args.epochs = 1
					args.output_dir = os.path.join(os.path.dirname(__file__), "..", "result", "EP2vec", args.cell_line, args.dataset)
					# if args.epochs != 10:
					# 	args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"×{args.ratio}", f"{args.k_list}_{args.stride}_{args.epochs}",f"{args.classifier}_{args.gbrt_tree_cnt}")
					ep2vec_stage2(args)

	# for dataset in dataset_list:
	# 	for cl in cell_line_list:
	# 		for classifier in ["GBRT"]:
	# 			for cv in ["random", "chromosomal"]:
	# 				for tree_cnt in [100, 1000, 4000]:
	# 					args.dataset = dataset
	# 					args.cell_line = cl
	# 					args.classifier = classifier
	# 					args.way_of_cv = cv
	# 					args.gbrt_tree_cnt = tree_cnt

	# 					if args.classifier == "GBRT":
	# 						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"{args.classifier}_{args.gbrt_tree_cnt}")
	# 					elif args.classifier == "KNN":
	# 						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"{args.classifier}_{args.knn_neighbor_cnt}")
	# 					elif args.classifier == "SVM":
	# 						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, args.classifier)
	# 					ep2vec_stage2(args)