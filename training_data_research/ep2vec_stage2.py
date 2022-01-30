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



def my_classifier(args):
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
	train_df = pd.read_csv(train_path)
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
		X_data.append(list(concat_vec.reshape(args.vector_size * 2)))
		Y_data.append(int(row_data["label"]))
		chrom_data.append(row_data["enhancer_chrom"])

	X_df = pd.DataFrame(
		data=X_data,
		columns=[f"x{i}" for i in range(args.vector_size * 2)]
	)
	X_df["chrom"] = chrom_data

	Y_df = pd.DataFrame(
		data=Y_data,
		columns=["label"]
	)

	print(X_df.head())
	print(Y_df.head())

	return X_df, Y_df


def my_cross_val(args, classifier, X_df, Y_df):

	if args.way_of_cv == "random":
		print("random 10-fold cross-validation...")

		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
		x = X_df.drop(columns="chrom").values
		y = Y_df["label"].values
		for fold, (train_index, test_index) in enumerate(cv.split(x, y)):
			output_path = os.path.join(args.output_dir, f"fold_{fold+1}")
			print(f"fold {fold+1}...")
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]

			classifier.fit(x_train, y_train) # 学習
			y_pred = classifier.predict_proba(x_test) # predict
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
	elif args.way_of_cv == "chromosomal":
		print("chromosomal-split cross-validateion...")

		test_chroms = [["chr1", "chr2"], ["chr3", "chr4"], ["chr5", "chr6"], ["chr7", "chr8"], ["chr9", "chr10"], ["chr11", "chr12"], ["chr13", "chr14"], ["chr15", "chr16"], ["chr17", "chr18"], ["chr19", "chr20"], ["chr21", "chr22"]]
		for fold, test_chrom in enumerate(test_chroms):
			output_path = os.path.join(args.output_dir, f"fold_{fold+1}")
			print(f"fold {fold+1}...")
			print(f"test chromosome : {test_chrom}")
			test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]
			test_index = X_df.query('chrom == @test_chrom1 or chrom == @test_chrom2').index.tolist()
			X_test_df = X_df.iloc[test_index, :]
			Y_test_df = Y_df.iloc[test_index, :]
			X_train_df = X_df.drop(index=test_index)
			Y_train_df = Y_df.drop(index=test_index)

			x_train = X_train_df.drop(columns="chrom").values
			y_train = Y_train_df["label"].values
			x_test = X_test_df.drop(columns="chrom").values
			y_test = Y_test_df["label"].values

			classifier.fit(x_train, y_train) # 学習
			y_pred = classifier.predict_proba(x_test) # predict
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

	d2v_path = ""
	input_dir = os.path.join(os.path.dirname(__file__), "ep2vec_d2v", args.cell_line, args.way_of_kmer)
	if args.way_of_kmer == "normal":
		d2v_path = os.path.join(input_dir, f"{args.k}_{args.stride}.d2v")
	elif args.way_of_kmer == "random":
		d2v_path = os.path.join(input_dir, f"{args.kmin}_{args.kmax}_{args.sentenceCnt}.d2v")

	d2v = Doc2Vec.load(d2v_path)
	os.system(f"mkdir -p {args.output_dir}")
	classifier = my_classifier(args)
	X_df, Y_df = classifier_preprocess(args, d2v)
	my_cross_val(args, classifier, X_df, Y_df)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k", help="k-merのk", type=int, default=6)
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--kmax", help="k-merのk", type=int, default=6)
	parser.add_argument("--kmin", help="k-merのk", type=int, default=3)
	parser.add_argument("--sentenceCnt", help="何個複製するか", type=int, default=3)
	parser.add_argument("--way_of_kmer", choices=["normal", "random"], default="random")
	parser.add_argument("--vector_size", help="分散表現の次元", type=int, default=100)
	parser.add_argument("--way_of_cv", help="染色体毎かランダムか", choices=["chromosomal", "random"], default="chromosomal")
	parser.add_argument("--classifier", type=str, choices=["GBRT", "KNN", "SVM"], default="GBRT", help="分類器に何を使うか")
	parser.add_argument("--gbrt_tree_cnt", type=int, default=1000, help="GBRTの木の数")
	parser.add_argument("--knn_neighbor_cnt", type=int, default=5, help="k近傍法の近傍数")
	args = parser.parse_args()

	dataset_list = ["new", "TargetFinder", "ep2vec"]
	cell_line_list = ["K562", "GM12878", "HUVEC", "HeLa-S3", "NHEK", "IMR90"]
	for dataset in dataset_list:
		for cl in cell_line_list:
			for classifier in ["GBRT"]:
				for tree_cnt in [100, 1000, 4000]:
					args.dataset = dataset
					args.cell_line = cl
					args.gbrt_tree_cnt = tree_cnt

					if args.classifier == "GBRT":
						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"{args.classifier}_{args.gbrt_tree_cnt}", "test")
					elif args.classifier == "KNN":
						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"{args.classifier}_{args.knn_neighbor_cnt}")
					elif args.classifier == "SVM":
						args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, args.classifier)
					ep2vec_stage2(args)