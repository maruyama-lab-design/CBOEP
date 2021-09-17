from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd

import os
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def create_extended_EnhPrm(args):
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
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
		print("bedfileからfastafileを作ります")
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

def make_extended_enhancer_promoter_model(args):

	tags = [] # doc2vec のための region tag を入れる
	sentences = [] # 塩基配列を入れる

	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		print(f"{cell_line} の エンハンサー 開始")
		enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
		fasta_lines = enhancer_fasta_file.readlines()

		id = 0
		for fasta_line in fasta_lines:
			# ">chr1:17000-18000" のような行はとばす.
			if fasta_line[0] == ">":
				continue
			else:
				tag = "ENHANCER_" + str(id)
				n_cnt = fasta_line.count("n")
				if n_cnt == 0: # "N" がある塩基配列をとばす
					tags.append(tag)
					sentences.append(make_kmer(6, 1, fasta_line))
				id += 1
		enhancer_fasta_file.close()
		print(f"{cell_line} の エンハンサー 終了")

		print(f"{cell_line} の プロモーター 開始")
		promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
		fasta_lines = promoter_fasta_file.readlines()

		id = 0
		for fasta_line in fasta_lines:
			# ">chr1:17000-18000" のような行はとばす.
			if fasta_line[0] == ">":
				continue
			else:
				tag = "PROMOTER_" + str(id)
				n_cnt = fasta_line.count("n")
				if n_cnt == 0:
					tags.append(tag)
					sentences.append(make_kmer(6, 1, fasta_line))
				id += 1
		promoter_fasta_file.close()
		print(f"{cell_line} の プロモーター 終了")

		# _____________________________________
		print(f"doc2vec 学習プロセス...")
		corpus = []
		for (tag, sentence) in zip(tags, sentences):
			corpus.append(TaggedDocument(sentence, [tag]))

		model = Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=8, epochs=10)
		model.build_vocab(corpus)
		model.train(
			corpus,
			total_examples=model.corpus_count,
			epochs=model.epochs
		)
		model.save(f"{args.my_data_folder_path}/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model")

def train_classifier(args):
	pass

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

	create_extended_EnhPrm(args)
	make_extended_enhancer_promoter_model(args)