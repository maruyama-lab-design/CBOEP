from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np

import os
import argparse

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def make_kmer(k, stride, sequence):
	#-----説明-----
	# sequence(塩基配列) を k-mer に区切り、sentence で返す
	# 返り値である sentence 内の k-mer 間は空白区切り
	#-------------

	length = len(sequence)
	sentence = ""
	start_pos = 0
	while start_pos <= length - k:
		# k-merに切る
		word = sequence[start_pos : start_pos + k]
		
		# 切り出したk-merを書き込む
		sentence += word + ' '

		start_pos += stride

	return sentence


def data_download(args, cell_line):

	# enhancer
	print("エンハンサーをダウンロードします.")
	os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/enhancers.bed -O {args.my_data_folder_path}bed/enhancer/{cell_line}_enhancers.bed")

	# promoter
	print("プロモーターをダウンロードします.")
	os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/promoters.bed -O {args.my_data_folder_path}bed/promoter/{cell_line}_promoters.bed")

	# reference genome
	print("リファレンスゲノムをダウンロードします.")
	os.system(f"wget {args.reference_genome_url} -O {args.my_data_folder_path}reference_genome/hg19.fa.gz")


def create_extended_EnhPrm(args, cell_line):
	print("エンハンサー, プロモーターのbedfileからfastafileを作成します.")

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# input bed
	extended_enhancer_bed_path = f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.bed"
	extended_promoter_bed_path = f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.bed"

	# output fasta
	extended_enhancer_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
	extended_promoter_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

	if not os.path.exists(extended_enhancer_bed_path):
		print("与えられたextendedエンハンサーのbedfileがありません")
		print("オリジナルのエンハンサーのbedfileから作成します...")
		text = ""
		with open(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed", "r") as origin_bed:
			lines = origin_bed.readlines()
			for line in lines:
				line = line.split("\t")
				chr, start_pos, end_pos = line[0], int(line[1]), int(line[2])
				start_pos -= args.E_extended_left_length
				end_pos += args.E_extended_right_length
				name = cell_line + "|" + chr + ":" + str(start_pos) + "-" + str(end_pos)
				text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n"
		with open(extended_enhancer_bed_path, "w") as extended_bed:
			extended_bed.write(text)
	if not os.path.exists(extended_promoter_bed_path):
		print("与えられたextendedプロモーターのbedfileがありません")
		print("オリジナルのプロモーターのbedfileから作成します...")
		text = ""
		with open(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed", "r") as origin_bed:
			lines = origin_bed.readlines()
			for line in lines:
				line = line.split("\t")
				chr, start_pos, end_pos = line[0], int(line[1]), int(line[2])
				start_pos -= args.P_extended_left_length
				end_pos += args.P_extended_right_length
				name = cell_line + "|" + chr + ":" + str(start_pos) + "-" + str(end_pos)
				text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n"
		with open(extended_promoter_bed_path, "w") as extended_bed:
			extended_bed.write(text)


	# bedtools で hg19 を bed 切り出し → fasta に保存
	print("fastafileを作ります")
	os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_enhancer_bed_path +" -fo "+ extended_enhancer_fasta_path)
	os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_promoter_bed_path +" -fo "+ extended_promoter_fasta_path)

	# 塩基配列を全て小文字へ
	seqs = ""
	with open(extended_enhancer_fasta_path, "r") as fout:
		seqs = fout.read()
	seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
	with open(extended_enhancer_fasta_path, "w") as fout:
		fout.write(seqs)
	
	with open(extended_promoter_fasta_path, "r") as fout:
		seqs = fout.read()
	seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
	with open(extended_promoter_fasta_path, "w") as fout:
		fout.write(seqs)


def make_extended_region_table(args, cell_line):
	# -----説明-----
	# 前提として、全領域の bedfile, fastafile が存在する必要があります.

		# enhancer のテーブルデータの例
			#	id				chr   	start	end		n_cnt
			#	ENHANCER_34		chr1	235686	235784	0

	# -------------

	print(f"全ての エンハンサー, プロモーター 領域について csvファイルを作成します.")
	print(f"{cell_line} 開始")
	print(f"エンハンサー...")
	enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")

	id = 0
	names = [] # chr1:900000-9100000 など
	region_ids = [] # ENHANCER_0 などの id を入れていく
	chrs = [] # chr1 などを入れていく
	starts = []	# start pos を入れていく
	ends = [] # end pos を入れていく
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = enhancer_fasta_file.readlines()

	for fasta_line in fasta_lines:

		# ">chr1:17000-18000" のような行.
		if fasta_line[0] == ">":
			region_id = "ENHANCER_" + str(id)
			region_ids.append(region_id)

			name = fasta_line[1:].replace("\n", "")
			chr, range_txt = name.split(":")[0], name.split(":")[1]
			start_pos, end_pos = range_txt.split("-")[0], range_txt.split("-")[1]
			names.append(name)
			chrs.append(chr)
			starts.append(start_pos)
			ends.append(end_pos)
		else:
			n_cnt = fasta_line.count("n")
			n_cnts.append(n_cnt)

			id += 1


	df = pd.DataFrame({
		"name":names,
		"id":region_ids,
		"chr":chrs,
		"start":starts,
		"end":ends,
		"n_cnt":n_cnts,
	})
	df.to_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.csv")

	enhancer_fasta_file.close()
	print(f"エンハンサー 完了")


	print(f"プロモーター...")
	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")

	id = 0
	names = [] # chr1:900000-9100000 など
	region_ids = [] # PROMOTER_0 などの id を入れていく
	chrs = [] # chr1 などを入れていく
	starts = []	# start pos を入れていく
	ends = [] # end pos を入れていく
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = promoter_fasta_file.readlines()

	for fasta_line in fasta_lines:

		# ">chr1:17000-18000" のような行.
		if fasta_line[0] == ">":
			region_id = "PROMOTER_" + str(id)
			region_ids.append(region_id)

			name = fasta_line[1:].replace("\n", "")
			chr, range_txt = name.split(":")[0], name.split(":")[1]
			start_pos, end_pos = range_txt.split("-")[0], range_txt.split("-")[1]
			names.append(name)
			chrs.append(chr)
			starts.append(start_pos)
			ends.append(end_pos)
		else:
			n_cnt = fasta_line.count("n")
			n_cnts.append(n_cnt)

			id += 1


	df = pd.DataFrame({
		"name":names,
		"id":region_ids,
		"chr":chrs,
		"start":starts,
		"end":ends,
		"n_cnt":n_cnts,
	})
	df.to_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv")

	promoter_fasta_file.close()
	print(f"プロモーター 完了")


def make_extended_enhancer_promoter_model(args, cell_line):
	global enhancers_num,promoters_num,positive_num,negative_num

	print("doc2vec のための前処理をします.")
	tags = [] # doc2vec のための region tag を入れる
	sentences = [] # 塩基配列を入れる

	print(f"{cell_line} の エンハンサー 開始")
	enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
	fasta_lines = enhancer_fasta_file.readlines()

	id = 0
	enhancers_num = 0
	for fasta_line in fasta_lines:
		# ">chr1:17000-18000" のような行はとばす.
		if fasta_line[0] == ">":
			continue
		else:
			tag = "ENHANCER_" + str(id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				tags.append(tag)
				sentences.append(make_kmer(args.k, args.stride, fasta_line))
			id += 1
	enhancer_fasta_file.close()
	print(f"{cell_line} の エンハンサー 終了")
	enhancers_num = id
	print(f"個数 {enhancers_num}")

	print(f"{cell_line} の プロモーター 開始")
	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
	fasta_lines = promoter_fasta_file.readlines()

	id = 0
	promoters_num = 0
	for fasta_line in fasta_lines:
		# ">chr1:17000-18000" のような行はとばす.
		if fasta_line[0] == ">":
			continue
		else:
			tag = "PROMOTER_" + str(id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				tags.append(tag)
				sentences.append(make_kmer(args.k, args.stride, fasta_line))
			id += 1
	promoter_fasta_file.close()
	print(f"{cell_line} の プロモーター 終了")
	promoters_num = id
	print(f"個数 {promoters_num}")

	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tags, sentences):
		corpus.append(TaggedDocument(sentence, [tag]))

	print(f"doc2vec 学習...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	model.build_vocab(corpus)
	model.train(
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("終了")
	model.save(f"{args.my_data_folder_path}/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model")


def make_training_txt(args, cell_line):
	global positive_num,negative_num

	print("トレーニングデータを参照してtxtfileを作成します.")
	positive_num = 0
	negative_num = 0
	
	enhancer_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.csv", usecols=["name", "id"])
	promoter_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv", usecols=["name", "id"])
	if not os.path.isfile(f"{args.my_data_folder_path}/train/{cell_line}_train.csv"):
		print("トレーニングデータが見つかりません. ダウンロードします.")
		os.system(f"wget https://raw.githubusercontent.com/wanwenzeng/ep2vec/master/{cell_line}rain.csv -O {args.my_data_folder_path}/train/{cell_line}_train.csv")
	train_csv = pd.read_csv(f"{args.my_data_folder_path}/train/GM12878_train.csv", usecols=["bin", "enhancer_name", "promoter_name", "label"])

	enhancer_names = enhancer_table["name"].to_list()
	enhancer_ids = enhancer_table["id"].to_list()
	promoter_names = promoter_table["name"].to_list()
	promoter_ids = promoter_table["id"].to_list()

	# ペア情報を training.txt にメモ
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


def training_classifier(args, cell_line):
	global positive_num, negative_num
	print("分類器を学習します.")
	print("doc2vecで学習したモデルをロード...")
	model = Doc2Vec.load(f'MyProject/data/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model')
	print("ロード完了")

	arrays = np.zeros((positive_num+negative_num, args.embedding_vector_dimention*2)) # X (従属変数 後に EnhとPrmの embedding vector が入る)
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
		enhancer_vec = model.dv[prefix_enhancer]
		promoter_vec = model.dv[prefix_promoter]
		# window_vec = window_model.docvecs[prefix_window]
		enhancer_vec = enhancer_vec.reshape((1,args.embedding_vector_dimention))
		promoter_vec = promoter_vec.reshape((1,args.embedding_vector_dimention))
		# window_vec = window_vec.reshape((1,self.vlen))
		arrays[i] = np.column_stack((enhancer_vec,promoter_vec))
		labels[i] = int(data[2])
		i = i + 1

	# 評価する指標
	score_funcs = ['f1', 'roc_auc', 'average_precision']
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	print("分類器学習中...")
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
	result.to_csv(f"{args.my_data_folder_path}/result/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.csv")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", default=5000)
	parser.add_argument("-E_extended_left_length", type=int, default=100)
	parser.add_argument("-E_extended_right_length", type=int, default=100)
	parser.add_argument("-P_extended_left_length", type=int, default=500)
	parser.add_argument("-P_extended_right_length", type=int, default=500)
	parser.add_argument("-embedding_vector_dimention", type=int, default=100)
	parser.add_argument("-k", type=int, default=6)
	parser.add_argument("-stride", type=int, default=1)
	args = parser.parse_args()


	for cell_line in args.cell_line_list:
		# bedfile のダウンロード
		# data_download(args, cell_line)

		# bedfile から fastafileを作成 
		create_extended_EnhPrm(args, cell_line)

		# enhancerとpromoterのtableデータを作成
		make_extended_region_table(args, cell_line)

		# doc2vec
		make_extended_enhancer_promoter_model(args, cell_line)

		# 分類器作成のためのトレーニングデータ情報を作成
		make_training_txt(args, cell_line)

		# 分類器学習, 評価
		training_classifier(args, cell_line)

