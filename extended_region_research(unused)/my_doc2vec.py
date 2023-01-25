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

def tagged_sentence_generator_unused(args, cell_line):
	# print("入力sequenceをファイルに書き出します．")

	enhancer_filename = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_editted.fa"
	promoter_filename = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_editted.fa"
	
	with open(enhancer_filename, "rt") as enh_fin, open(promoter_filename, "rt") as prm_fin:
		for fin in [enh_fin, prm_fin]:
			for record in SeqIO.parse(fin, "fasta"):
				paragraph_tag = record.id.split("~")[0]
				region_seq = str(record.seq)
				if args.way_of_kmer == "normal": #固定長k-mer
					region_seq_kmer_list = utils.make_kmer_list(args.k, args.stride, region_seq)
					yield TaggedDocument(region_seq_kmer_list, [paragraph_tag])
				elif args.way_of_kmer == "random": #ランダム長k-mer
					for _ in range(args.sentence_cnt):
						region_seq_kmer_list = utils.make_random_kmer_list(args.k_min, args.k_max, region_seq)
						yield TaggedDocument(region_seq_kmer_list, [paragraph_tag])

class TaggedSentencesIterator_unused():
	# 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
	def __init__(self, generator_function, args, cell_line):
		self.generator_function = generator_function
		self.args = args
		self.cell_line = cell_line
		self.generator = self.generator_function(self.args, self.cell_line)

	def __iter__(self):
		# reset the generator
		self.generator = self.generator_function(self.args, self.cell_line)
		return self

	def __next__(self):
		result = next(self.generator)
		if result is None:
			raise StopIteration
		else:
			return result

def make_input_txt_for_doc2vec(args, cell_line):
	print("入力sequenceをファイルに書き出します．")

	enhancer_filename = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_editted.fa"
	promoter_filename = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_editted.fa"

	
	with open("input_for_d2v.txt", "w") as fout:
		with open(enhancer_filename, "rt") as enh_fin, open(promoter_filename, "rt") as prm_fin:
			for fin in [enh_fin, prm_fin]:
				for record in SeqIO.parse(fin, "fasta"):
					paragraph_tag = record.id
					region_seq = str(record.seq)
					if args.way_of_kmer == "normal": #固定長k-mer
						region_seq_kmer_list = utils.make_kmer_list(args.k, args.stride, region_seq)
						fout.write(paragraph_tag + "\t" + "\t".join(region_seq_kmer_list) + "\n")
					elif args.way_of_kmer == "random": #ランダム長k-mer
						for _ in range(args.sentence_cnt): # 100 がギリ
							region_seq_kmer_list = utils.make_random_kmer_list(args.k_min, args.k_max, region_seq)
							fout.write(paragraph_tag + "\t" + "\t".join(region_seq_kmer_list) + "\n")

def tagged_sentence_generator(filename):
	with open(filename, "r") as fin: # file読み込み
		for line in fin:
			tag_and_sentence = line.split()
			yield TaggedDocument(tag_and_sentence[1:], [tag_and_sentence[0]])

class TaggedSentencesIterator():
	# これは正しく動いてそう　epochごとにシャッフルはしてない？
	# 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
	def __init__(self, generator_function, filename):
		self.generator_function = generator_function
		self.filename = filename
		self.generator = self.generator_function(self.filename)

	def __iter__(self):
		# reset the generator
		self.generator = self.generator_function(self.filename)
		return self

	def __next__(self):
		result = next(self.generator)
		if result is None:
			raise StopIteration
		else:
			return result


def make_paragraph_vector_from_enhancer_and_promoter_using_iterator(args, cell_line): # イテレータを使ってメモリの節約を試みる

	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")
	make_input_txt_for_doc2vec(args, cell_line) # 前処理(ファイル書き出し)
	# _____________________________________
	print(f"doc2vec training...")
	tagged_sentence_iter = TaggedSentencesIterator(tagged_sentence_generator, "input_for_d2v.txt")
	model = Doc2Vec(
		tagged_sentence_iter,
		min_count=1,
		window=10,
		vector_size=args.embedding_vector_dimention,
		sample=1e-4,
		negative=5,
		workers=8,
		epochs=10
	)
	print("doc2vec 終了")
	d2v_model_path = os.path.join(args.my_data_folder_path, "d2v", f"{args.output}.d2v")
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
				paragraph_tag = record.id
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