# classifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics

from gensim.models import Doc2Vec

import os
import pandas as pd 
import numpy as np 
import argparse



def get_classifier(args):
	if args.classifier == "GBRT":
		return GradientBoostingClassifier(n_estimators = args.gbrt_tree_cnt, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	elif args.classifier == "KNN":
		return KNeighborsClassifier(n_neighbors=args.knn_neighbor_cnt) # k近傍法
	elif args.classifier == "SVM":
		return svm.SVC(kernel='linear', random_state=None, probability=True)


def classifier_preprocess(args, d2v):

	chrom_data = []
	X_data = [] # ２次元配列
	Y_data = [] # １次元配列

	paragraphTag_list = list(d2v.dv.index_to_key)

	train_path = os.path.join(os.path.dirname(__file__), "training_data", args.dataset, f"{args.cell_line}_train.csv")
	if args.dataset == "new":
		train_path = os.path.join(os.path.dirname(__file__), "training_data", args.dataset, f"×{args.ratio}", f"{args.cell_line}_train.csv")
	train_df = pd.read_csv(train_path, usecols=["enhancer_name", "enhancer_name", "promoter_name", "label"])
	data_list = []
	for _, row_data in train_df.iterrows():
		enhName = row_data["enhancer_name"]
		prmName = row_data["promoter_name"]
		if (enhName not in paragraphTag_list) or (prmName not in paragraphTag_list):
			continue
		enhVec = d2v.dv[enhName]
		prmVec = d2v.dv[prmName]
		enhVec = enhVec.reshape((1,args.vector_size))
		prmVec = prmVec.reshape((1,args.vector_size))
		concat_vec = np.column_stack((enhVec,prmVec)) # concat
		data = list(concat_vec.reshape(args.vector_size * 2))
		data.append(int(row_data["label"]))
		data_list.append(data) 
		chrom_data.append(row_data["enhancer_chrom"])

	df = pd.DataFrame(
		data=data_list,
		columns=[f"x{i}" for i in range(args.vector_size * 2)] + ["label"]
	)
	df["enhancer_chrom"] = chrom_data

	return df


def get_weights(y):

	weights_dic = {
		0: 1 / (np.sum(y==0) / len(y)), # 負例重み
		1: 1 / (np.sum(y==1) / len(y)) # 正例重み
	}

	weights_arr = np.zeros(len(y))

	for i in range(len(y)):
		weights_arr[i] = weights_dic[y[i]]

	return weights_arr


def chromosomal_cv(args, df):
	
	print("chromosomal-split cross-validateion...")

	test_chroms = [["chr1", "chr2"], ["chr3", "chr4"], ["chr5", "chr6"], ["chr7", "chr8"], ["chr9", "chr10"], ["chr11", "chr12"], ["chr13", "chr14"], ["chr15", "chr16"], ["chr17", "chr18"], ["chr19", "chr20"], ["chr21", "chr22"]]
	for fold, test_chrom in enumerate(test_chroms):
		output_path = os.path.join(args.output_dir, f"fold_{fold+1}")
		print(f"fold {fold+1}...")
		print(f"test chromosome : {test_chrom}")
		test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]
		
		test_index = df.query('enhancer_chrom == @test_chrom1 or enhancer_chrom == @test_chrom2').index.tolist()
		test_df = df.iloc[test_index, :]
		train_df = df.drop(index=test_index)

		x_train = train_df.drop(columns="enhancer_chrom").values
		y_train = train_df["label"].values
		weights = get_weights(y_train)
		classifier = get_classifier(args)
		classifier.fit(x_train, y_train, sample_weight=weights) # 学習

		x_test = test_df.drop(columns="enhancer_chrom").values
		y_test = test_df["label"].values
		y_pred = classifier.predict_proba(x_test) # 予測
		y_pred = [prob[1] for prob in y_pred] # 正例確率のみを抽出
		print(y_pred)

		result_df = pd.DataFrame(
			{
				"y_test": y_test,
				"y_pred": y_pred
			},
			index=None
		)
		result_df.to_csv(output_path)


def ep2vec_stage2(args):
	os.system(f"mkdir -p {args.output_dir}")

	input_dir = os.path.join(os.path.dirname(__file__), "ep2vec_d2v", args.cell_line, args.way_of_kmer)
	d2v_path = os.path.join(input_dir, f"{args.k_list}_{args.stride}.d2v")
	d2v = Doc2Vec.load(d2v_path)
	df = classifier_preprocess(args, d2v)
	chromosomal_cv(args, df)




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

	dataset_list = ["new", "TargetFinder", "ep2vec"]
	cell_line_list = ["GM12878", "K562", "NHEK", "HeLa-S3", "IMR90", "HUVEC"]

	args.dataset = "new"
	args.classifier = "GBRT"
	args.gbrt_tree_cnt = 4000
	args.way_of_cv = "chromosomal"
	# k_mer_set = ["1", "2", "3", "4", "5", "6", "1,2,3,4,5,6", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6"]
	k_mer_set = ["6"]
	# k_mer_set = ["1,2,3,4,5,6,7,8", "7", "8"]
	args.stride = 1
	for cl in cell_line_list:
		for ratio in [1]:
			for k_list in k_mer_set:
				args.cell_line = cl
				args.ratio = ratio
				args.k_list = k_list
				args.epochs = 1
				args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"×{args.ratio}", f"{args.k_list}_{args.stride}",f"{args.classifier}_{args.gbrt_tree_cnt}")
				if args.epochs != 10:
					args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"×{args.ratio}", f"{args.k_list}_{args.stride}_{args.epochs}",f"{args.classifier}_{args.gbrt_tree_cnt}")
				ep2vec_stage2_v2(args)

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