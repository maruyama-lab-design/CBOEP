from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse

from Bio import SeqIO
import gzip

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 自作関数
import utils

def make_input_txt_for_doc2vec(args, cell_line):
    enhancer_filename = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_editted.fa"
    promoter_filename = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_editted.fa"

	
    with open("input_for_d2v.txt", "w") as fout:
        with open(enhancer_filename, "rt") as enh_fin, open(promoter_filename, "rt") as prm_fin:
            for fin in [enh_fin, prm_fin]:
                for record in SeqIO.parse(fin, "fasta"):
                    paragraph_tag = record.id.split("~")[0]
                    region_seq = str(record.seq)
                    if args.way_of_kmer == "normal": #固定長k-mer
                        fout.write(paragraph_tag + "\t" + region_seq + "\n")
                    elif args.way_of_kmer == "random": #ランダム長k-mer
                        for _ in range(10): # 100 がギリ
                            fout.write(paragraph_tag + "\t" + region_seq + "\n")


class CorporaIterator():
	def __init__(self, args, fileobject):
		self.args = args
		self.fileobject = fileobject

	def __iter__(self):
		return self

	def __next__(self):
		tag_and_sequence = next(self.fileobject)
		if tag_and_sequence is None:
			raise StopIteration
		else:
			paragraph_tag, sequence = tag_and_sequence.split()
			if self.args.way_of_kmer == "normal":
				return TaggedDocument(utils.make_kmer_list(self.args.k, self.args.stride, sequence), [paragraph_tag])
			elif self.args.way_of_kmer == "random":
				return TaggedDocument(utils.make_random_kmer_list(self.args.k_min, self.args.k_max, sequence), [paragraph_tag])


def make_paragraph_vector_from_enhancer_and_promoter_using_iterator(args, cell_line): # イテレータを使ってメモリの節約を試みる
	make_input_txt_for_doc2vec(args, cell_line) # 前処理

	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")
	# _____________________________________
	print(f"doc2vec training...")
	with open("input_for_d2v.txt") as fileobject:
		corpus_iter = CorporaIterator(args, fileobject)
		model = Doc2Vec(documents=corpus_iter, min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	print("doc2vec 終了")
	d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
	model.save(d2v_model_path)


def make_paragraph_vector_from_enhancer_and_promoter(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	sentence_list = [] # numpy にしてみる
	paragraph_tag_list = []

	for region_type in ["enhancer", "promoter"]:
		print(f"{region_type} doc2vec preprocessing...")
		input_fasta_path = os.path.join(args.my_data_folder_path, "fasta", region_type, f"{cell_line}_{region_type}s_editted.fa")
		with open(input_fasta_path, "rt") as fin: # fastafile の読みこみ
			for record in SeqIO.parse(fin, "fasta"):
				paragraph_tag = record.id.split("~")[0]
				region_seq = str(record.seq)

				if args.way_of_kmer == "normal": #固定長k-mer
					paragraph_tag_list.append(paragraph_tag)
					sentence_list.append(utils.make_kmer_list(args.k, args.stride, region_seq))
				elif args.way_of_kmer == "random": #ランダム長k-mer
					for _ in range(args.sentence_cnt): # 100 がギリ
						paragraph_tag_list.append(paragraph_tag)
						sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, region_seq))
	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(paragraph_tag_list, sentence_list):
		corpus.append(TaggedDocument(sentence, [tag])) # doc2vec前のsentenceへのtagつけ

	print(f"doc2vec training...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	model.build_vocab(corpus) # 単語の登録
	model.train( # ここで学習開始
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("doc2vec 終了")
	d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")
	model.save(d2v_model_path)



def make_paragraph_vector_from_enhancer_and_promoter_unused(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	sentence_list = []
	tag_list = []

	for region_type in ["enhancer", "promoter"]:
		print(f"{region_type}...")

		bed_df = pd.read_csv(f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed.csv", usecols=["name_origin"])
		with open(f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s.fa", "rt") as fin: # fastafile の読みこみ
			for record in SeqIO.parse(fin, "fasta"):
				fasta_region_name = record.id.split("::")[0].replace("\n", "") # region name を fastafileから取得
				region_seq = str(record.seq)

				region_index = bed_df[bed_df["name_origin"] == fasta_region_name].index.tolist()[0] # regionはbed_dfの何番目のindexか
				tag = region_type + "_" + str(region_index) # enhancer_0 など
				if args.way_of_kmer == "normal":
					tag_list.append(tag)
					sentence_list.append(utils.make_kmer_list(args.k, args.stride, region_seq))
				elif args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt):
						tag_list.append(tag)
						sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, region_seq))
	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tag_list, sentence_list):
		corpus.append(TaggedDocument(sentence, [tag])) # doc2vec前のsentenceへのtagつけ

	print(f"doc2vec 学習...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	model.build_vocab(corpus) # 単語の登録
	model.train(
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("終了")
	model.save(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")


def make_paragraph_vector_from_enhancer_and_promoter_unused2(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	sentence_list = []
	tag_list = []

	for region_type in ["enhancer", "promoter"]:
		print(f"{region_type}...")
		region_missing_data_cnt = 0

		bed_df = pd.read_csv(f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed.csv", usecols=["name_origin"])
		with open(f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s.fa", "rt") as fin: # fastafile の読みこみ
			for record in SeqIO.parse(fin, "fasta"):
				fasta_region_name = record.id.split("::")[0].replace("\n", "")
				region_seq = str(record.seq)
				region_complement_seq = str(record.seq.complement())

				if region_seq.count("n") > 0: # 配列にnが含まれていたら除く
					# n がある物を省く関数を用意
					region_missing_data_cnt += 1
					continue

				region_index = bed_df[bed_df["name_origin"] == fasta_region_name].index.tolist()[0]
				tag = region_type + "_" + str(region_index)
				if args.way_of_kmer == "normal":
					tag_list.append(tag)
					sentence_list.append(utils.make_kmer_list(args.k, args.stride, region_seq))
					tag_list.append(tag)
					sentence_list.append(utils.make_kmer_list(args.k, args.stride, region_complement_seq))
				elif args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt):
						tag_list.append(tag)
						sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, region_seq))
						tag_list.append(tag)
						sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, region_complement_seq))

		print(f"{region_type} 全 {len(bed_df)}個 中 {region_missing_data_cnt}個 が学習から除外されました")
	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tag_list, sentence_list):
		corpus.append(TaggedDocument(sentence, [tag])) # doc2vec前のsentenceへのtagつけ

	print(f"doc2vec 学習...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	model.build_vocab(corpus) # 単語の登録
	model.train(
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("終了")
	model.save(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")


def make_paragraph_vector_from_enhancer_and_promoter_unused3(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	print(f"エンハンサー...")
	enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
	enhancer_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}_r.fa", "r")
	fasta_lines = enhancer_fasta_file.readlines()
	reverse_fasta_lines = enhancer_reverse_fasta_file.readlines()

	tags = [] # doc2vec のための region tag を入れる
	sentences = [] # 塩基配列を入れる (corpus)

	enhancer_id = 0
	for (fasta_line, reverse_fasta_line) in zip(fasta_lines, reverse_fasta_lines):

		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行はとばす.
			continue
		else:
			tag = "ENHANCER_" + str(enhancer_id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				if args.way_of_kmer == "normal":
					tags.append(tag)
					sentences.append(utils.make_kmer_list(args.k, args.stride, fasta_line))
					tags.append(tag) # reverse complement用に２回
					sentences.append(utils.make_kmer_list(args.k, args.stride, reverse_fasta_line))
				if args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt): # make N random k-mer sentence
						tags.append(tag)
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, fasta_line))
						tags.append(tag) # reverse complement用に２回
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, reverse_fasta_line))
			enhancer_id += 1

	enhancer_fasta_file.close()
	enhancer_reverse_fasta_file.close()
	print(f"エンハンサー 終了")
	enhancer_cnt = len(sentences) // (2 * args.sentence_cnt)
	print(f"エンハンサーの個数 {enhancer_cnt}")


	print(f"プロモーター...")
	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
	promoter_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}_r.fa", "r")
	fasta_lines = promoter_fasta_file.readlines()
	reverse_fasta_lines = promoter_reverse_fasta_file.readlines()

	promoter_id = 0
	for (fasta_line, reverse_fasta_line) in zip(fasta_lines, reverse_fasta_lines):
		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行はとばす.
			continue
		else:
			tag = "PROMOTER_" + str(promoter_id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				if args.way_of_kmer == "normal":
					tags.append(tag)
					sentences.append(utils.make_kmer_list(args.k, args.stride, fasta_line))
					tags.append(tag) # reverse complement用に２回
					sentences.append(utils.make_kmer_list(args.k, args.stride, reverse_fasta_line))
				if args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt):
						tags.append(tag)
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, fasta_line))
						tags.append(tag) # reverse complement用に２回
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, reverse_fasta_line))
			promoter_id += 1

	promoter_fasta_file.close()
	promoter_reverse_fasta_file.close()
	print(f"{cell_line} の プロモーター 終了")
	print(f"プロモーターの個数 {(len(sentences) // (2 * args.sentence_cnt)) - enhancer_cnt}")

	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tags, sentences):
		corpus.append(TaggedDocument(sentence, [tag])) # doc2vec前のsentenceへのtagつけ

	print(f"doc2vec 学習...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=8, epochs=10)
	model.build_vocab(corpus) # 単語の登録
	model.train(
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("終了")
	model.save(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")


def make_paragraph_vector_from_enhancer_only(args, cell_line):
	print(f"{cell_line} のエンハンサーのみに対しdoc2vecで学習します")
	print("doc2vec のための前処理をします.")

	enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
	enhancer_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_r_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
	fasta_lines = enhancer_fasta_file.readlines()
	reverse_fasta_lines = enhancer_reverse_fasta_file.readlines()

	tags = [] # doc2vec のための region tag を入れる
	sentences = [] # 塩基配列を入れる (corpus)
	enhancer_id = 0
	for (fasta_line, reverse_fasta_line) in zip(fasta_lines, reverse_fasta_lines):
		# ">chr1:17000-18000" のような行はとばす.
		if fasta_line[0] == ">":
			continue
		else:
			tag = "ENHANCER_" + str(enhancer_id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				if args.way_of_kmer == "normal":
					tags.append(tag)
					sentences.append(utils.make_kmer_list(args.k, args.stride, fasta_line))
					tags.append(tag) # reverse complement用に２回
					sentences.append(utils.make_kmer_list(args.k, args.stride, reverse_fasta_line))
				if args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt):
						tags.append(tag)
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, fasta_line))
						tags.append(tag) # reverse complement用に２回
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, reverse_fasta_line))
			enhancer_id += 1

	enhancer_fasta_file.close()
	enhancer_reverse_fasta_file.close()
	print(f"{cell_line} の エンハンサー 終了")
	print(f"エンハンサーの個数 {len(sentences) // (2 * args.sentence_cnt)}")
	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tags, sentences):
		corpus.append(TaggedDocument(sentence, [tag]))

	print(f"doc2vec 学習...")
	model = Doc2Vec(min_count=1, window=10, vector_size=args.embedding_vector_dimention, sample=1e-4, negative=5, workers=30, epochs=10)
	model.build_vocab(corpus)
	model.train(
		corpus,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	print("終了")
	model.save(f"{args.my_data_folder_path}/d2v/{cell_line},el={args.E_extended_left_length},er={args.E_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")


def make_paragraph_vector_from_promoter_only(args, cell_line):
	print(f"{cell_line} のプロモーターのみに対しdoc2vecで学習します")
	print("doc2vec のための前処理をします.")

	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
	promoter_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_r_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
	fasta_lines = promoter_fasta_file.readlines()
	reverse_fasta_lines = promoter_reverse_fasta_file.readlines()

	tags = [] # doc2vec のための region tag を入れる
	sentences = [] # 塩基配列を入れる (corpus)
	promoter_id = 0
	for (fasta_line, reverse_fasta_line) in zip(fasta_lines, reverse_fasta_lines):
		# ">chr1:17000-18000" のような行はとばす.
		if fasta_line[0] == ">":
			continue
		else:
			tag = "PROMOTER_" + str(promoter_id)
			n_cnt = fasta_line.count("n")
			if n_cnt == 0: # "N" を含むような配列は学習に使わない
				if args.way_of_kmer == "normal":
					tags.append(tag)
					sentences.append(utils.make_kmer_list(args.k, args.stride, fasta_line))
					tags.append(tag) # reverse complement用に２回
					sentences.append(utils.make_kmer_list(args.k, args.stride, reverse_fasta_line))
				if args.way_of_kmer == "random":
					for _ in range(args.sentence_cnt):
						tags.append(tag)
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, fasta_line))
						tags.append(tag) # reverse complement用に２回
						sentences.append(utils.make_random_kmer_list(args.k_min, args.k_max, reverse_fasta_line))
			promoter_id += 1

	promoter_fasta_file.close()
	promoter_reverse_fasta_file.close()
	print(f"{cell_line} の プロモーター 終了")
	print(f"プロモーターの個数 {len(sentences) // (2 * args.sentence_cnt)}")

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
	model.save(f"{args.my_data_folder_path}/d2v/{cell_line},pl={args.P_extended_left_length},pr={args.P_extended_right_length},kmer={args.way_of_kmer},N={args.sentence_cnt}.d2v")