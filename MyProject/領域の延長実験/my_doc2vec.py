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

def make_paragraph_vector_from_enhancer_and_promoter(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	sentence_list = []
	tag_list = []
	record_dict = {}
	with gzip.open(f"{args.my_data_folder_path}/reference_genome/hg19.fa.gz", "rt") as handle:
		record_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta")) # これで染色体名がkeyになります.

	for region_type in ["enhancer", "promoter"]:
		print(f"{region_type}...")
		region_missing_data_cnt = 0

		extended_region_df = pd.read_csv(f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed.csv", usecols=["chrom", "start_extended", "end_extended"])
		for index, row_data in extended_region_df.iterrows():
			tag = region_type + "_" + str(index)
			seq_by_chrom = str(record_dict[row_data["chrom"]].seq)
			complement_seq_by_chrom = str(record_dict[row_data["chrom"]].seq.complement())
			target_seq = seq_by_chrom[row_data["start_extended"] : row_data["end_extended"]].lower()
			target_complement_seq = complement_seq_by_chrom[row_data["start_extended"] : row_data["end_extended"]].lower()
			# print(tag)
			# print(target_seq)
			# print(target_complement_seq)
			# foo = ""

			if target_seq.count("n") >= 0:
				region_missing_data_cnt += 1
				continue


			if args.way_of_kmer == "normal":
				tag_list.append(tag)
				sentence_list.append(utils.make_kmer_list(args.k, args.stride, target_seq))
				tag_list.append(tag)
				sentence_list.append(utils.make_kmer_list(args.k, args.stride, target_complement_seq))
			elif args.way_of_kmer == "random":
				for _ in range(args.sentence_cnt):
					tag_list.append(tag)
					sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, target_seq))
					tag_list.append(tag)
					sentence_list.append(utils.make_random_kmer_list(args.k_min, args.k_max, target_complement_seq))

		print(f"{region_type} 全 {len(extended_region_df)}個 中 {region_missing_data_cnt}個 が学習から除外されました")
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


def make_paragraph_vector_from_enhancer_and_promoter_unused(args, cell_line):
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