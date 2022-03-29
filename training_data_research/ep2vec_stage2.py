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
	if args.dataset == "new":
		train_path = os.path.join(os.path.dirname(__file__), "training_data", args.dataset, f"×{args.ratio}", f"{args.cell_line}_train.csv")
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


def get_weights(args, y):

	weights = {
		0: 1 / (np.sum(y==0) / len(y)),
		1: 1 / (np.sum(y==1) / len(y))
	}

	w_df = pd.DataFrame(np.zeros((len(y), 1)), columns=["weight"]) 

	for i in range(len(y)):
		w_df.loc[i, "weight"] = weights[y[i]]
	# w_df.loc[y["label"]==0, "weight"] = weights[0]
	# w_df.loc[y["label"]==1, "weight"] = weights[1]

	# print(np.sum(y==1))

	return w_df["weight"].values


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
			weights = get_weights(args, y_train)
			classifier.fit(x_train, y_train, sample_weight=weights) # 学習
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
			weights = get_weights(args, y_train)
			classifier.fit(x_train, y_train, sample_weight=weights) # 学習
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


def getF1(y_true, y_prob, threshold=0.5):
	y_pred = [1 if i >= threshold else 0 for i in y_prob]
	# print(f1_score(y_true, y_pred))
	print(f1_score(y_true, y_pred))
	return f1_score(y_true, y_pred)


def show_averageF1_in_allFold(resultDir_path):
	files = os.listdir(resultDir_path)
	files_file = [f for f in files if os.path.isfile(os.path.join(resultDir_path, f))]
	# print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
	F1_score = np.zeros(len(files_file))
	for i, result_file in enumerate(files_file):
		# print(result_file)
		df = pd.read_csv(os.path.join(resultDir_path, result_file))
		y_true = df["y_test"].tolist()
		y_prob = df["y_pred"].tolist()
		F1_score[i] = getF1(y_true, y_prob)

	print({"mean": np.mean(F1_score), "yerr": np.std(F1_score)/math.sqrt(len(files_file))})



def ep2vec_stage2_v2(args):

	d2v_path = ""
	input_dir = os.path.join(os.path.dirname(__file__), "ep2vec_d2v", args.cell_line, args.way_of_kmer)
	if args.way_of_kmer == "normal":
		d2v_path = os.path.join(input_dir, f"{args.k_list}_{args.stride}.d2v")
	elif args.way_of_kmer == "random":
		# d2v_path = os.path.join(input_dir, f"{args.kmin}_{args.kmax}_{args.sentenceCnt}.d2v")
		print("エラー！！")

	d2v = Doc2Vec.load(d2v_path)
	os.system(f"mkdir -p {args.output_dir}")
	classifier = my_classifier(args)
	X_df, Y_df = classifier_preprocess(args, d2v)
	my_cross_val(args, classifier, X_df, Y_df)

	# show_averageF1_in_allFold(args.output_dir)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--ratio", default=5)
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k", help="k-merのk", type=int, default=6)
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
	cell_line_list = ["GM12878"]

	args.dataset = "new"
	args.classifier = "GBRT"
	args.gbrt_tree_cnt = 4000
	args.way_of_cv = "chromosomal"
	# k_mer_set = ["1", "2", "3", "4", "5", "6", "1,2,3,4,5,6", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6"]
	k_mer_set = ["6"]
	args.stride = 1
	for cl in cell_line_list:
		for k_list in k_mer_set:
			args.cell_line = cl
			args.k_list = k_list
			args.output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_result", args.dataset, args.cell_line, args.way_of_cv, f"{args.k_list}_{args.stride}",f"{args.classifier}_{args.gbrt_tree_cnt}")
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