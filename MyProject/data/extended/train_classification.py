import os
import pandas as pd
import numpy as np
import argparse
from gensim.models.doc2vec import Doc2Vec

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier

def make_training_txt(args):
	global positive_num,negative_num
	positive_num = 0
	negative_num = 0
	
	enhancer_table = pd.read_csv("MyProject/data/table/region/enhancer/GM12878_enhancers_100_100.csv", usecols=["name", "id"])
	promoter_table = pd.read_csv("MyProject/data/table/region/promoter/GM12878_promoters_100_100.csv", usecols=["name", "id"])
	# os.system(f"wget https://raw.githubusercontent.com/wanwenzeng/ep2vec/master/GM12878train.csv -O {args.my_data_folder_path}/train/GM12878_train.csv")
	train_csv = pd.read_csv(f"{args.my_data_folder_path}/train/GM12878_train.csv", usecols=["bin", "enhancer_name", "promoter_name", "label"])

	enhancer_names = enhancer_table["name"].to_list()
	enhancer_ids = enhancer_table["id"].to_list()
	promoter_names = promoter_table["name"].to_list()
	promoter_ids = promoter_table["id"].to_list()

	# ペア情報を training.txt にメモしたい
	fout = open('training.txt','w')
	for _, data in train_csv.iterrows(): # train.csv を1行ずつ読み込み
		enhancer_id = "nan"
		promoter_id = "nan"
		
		train_enhancer_name = data["enhancer_name"].split("|")[1]
		chr, enhancer_range = train_enhancer_name.split(":")[0], train_enhancer_name.split(":")[1]
		train_enhancer_start, train_enhancer_end = int(enhancer_range.split("-")[0]) - args.E_extended_left_length, int(enhancer_range.split("-")[1]) + args.E_extended_right_length
		train_enhancer_name = chr + ":" + str(train_enhancer_start) + "-" + str(train_enhancer_end)

		train_promoter_name = data["promoter_name"].split("|")[1]
		chr, promoter_range = train_promoter_name.split(":")[0], train_promoter_name.split(":")[1]
		train_promoter_start, train_promoter_end = int(promoter_range.split("-")[0]) - args.P_extended_left_length, int(promoter_range.split("-")[1]) + args.P_extended_right_length
		train_promoter_name = chr + ":" + str(train_promoter_start) + "-" + str(train_promoter_end)

		# print(train_enhancer_name, train_promoter_name)
		if train_enhancer_name in enhancer_names:
			enhancer_id = enhancer_ids[enhancer_names.index(train_enhancer_name)] # enhancer名から何番目のenhancerであるかを調べる
		if train_promoter_name in promoter_names:
			promoter_id = promoter_ids[promoter_names.index(train_promoter_name)] # promoter名から何番目のpromoterであるかを調べる
		
		if enhancer_id == "nan" or promoter_id == "nan":
			continue
		label = str(data["label"])

		# enhancer の ~ 番目と promoter の ~ 番目 は pair/non-pair であるというメモを書き込む
		fout.write(str(enhancer_id)+'\t'+str(promoter_id)+'\t'+label+'\n')

		if label == '1': # 正例
			positive_num = positive_num + 1
		else: # 負例
			negative_num = negative_num + 1

	print(f"正例: {positive_num}")
	print(f"負例: {negative_num}")


def train(args):
	global positive_num, negative_num
	for cell_line in args.cell_line_list:
		model = Doc2Vec.load(f'MyProject/data/model/{cell_line}_enhancer_100_100_promoter_100_100.model')

		arrays = np.zeros((positive_num+negative_num, 100*2)) # X (従属変数 後に EnhとPrmの embedding vector が入る)
		labels = np.zeros(positive_num+negative_num) # Y (目的変数 後に ペア情報{0 or 1}が入る)
		num    = positive_num+negative_num

		# 分類器を用意
		estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
		# メモしておいたペア情報を使う
		fin = open('training.txt','r')
		i = 0
		for line in fin:
			data = line.strip().split()
			prefix_enhancer = data[0] # "ENHANCER_0" などのembedding vector タグ
			prefix_promoter = data[1] # "PROMOTER_0" などのembedding vector タグ
			# prefix_window = 'WINDOW_' + data[2] # "PROMOTER_0" などのembedding vector タグ
			enhancer_vec = model.docvecs[prefix_enhancer]
			promoter_vec = model.docvecs[prefix_promoter]
			# window_vec = window_model.docvecs[prefix_window]
			enhancer_vec = enhancer_vec.reshape((1,100))
			promoter_vec = promoter_vec.reshape((1,100))
			# window_vec = window_vec.reshape((1,self.vlen))
			arrays[i] = np.column_stack((enhancer_vec,promoter_vec))
			labels[i] = int(data[2])
			i = i + 1

		# 評価する指標
		score_funcs = ['f1', 'roc_auc', 'average_precision']
		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
		scores = cross_validate(estimator, arrays, labels, scoring = score_funcs, cv = cv, n_jobs = -1)

		# 得られた指標を出力する
		print('f1:', scores['test_f1'].mean())
		print('auROC:', scores['test_roc_auc'].mean())
		print('auPRC:', scores['test_average_precision'].mean())
		f1 = scores['test_f1']
		auROC = scores['test_roc_auc']
		auPRC =  scores['test_average_precision']
		result = pd.DataFrame(
			{
			"f1": f1,
			"auROC": auROC,
			"auPRC": auPRC,
			},
			index = ["1-fold", "2-fold", "3-fold", "4-fold", "5-fold", "6-fold", "7-fold", "8-fold", "9-fold", "10-fold"]	
		)
		result.to_csv(f"{args.my_data_folder_path}/result/{cell_line}_enhancer_100_100_promoter_100_100.csv")		



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="未完成")
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", default=5000)
	parser.add_argument("-E_extended_left_length", type=int, default=100)
	parser.add_argument("-E_extended_right_length", type=int, default=100)
	parser.add_argument("-P_extended_left_length", type=int, default=100)
	parser.add_argument("-P_extended_right_length", type=int, default=100)
	args = parser.parse_args()

	make_training_txt(args)
	train(args)
