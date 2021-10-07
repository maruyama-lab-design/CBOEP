from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 自作関数
from utils import make_kmer_list


def make_paragraph_vector_from_enhancer_and_promoter(args, cell_line):
	print(f"{cell_line} のエンハンサーとプロモーターの両方を1つのdoc2vecで学習します")
	print("doc2vec のための前処理 開始")

	print(f"{cell_line} の エンハンサー...")
	enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
	enhancer_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_r_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")
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
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, fasta_line))
				tags.append(tag) # reverse complement用に２回
				sentences.append(make_kmer_list(args.k, args.stride, reverse_fasta_line))
			enhancer_id += 1

	enhancer_fasta_file.close()
	enhancer_reverse_fasta_file.close()
	print(f"{cell_line} の エンハンサー 終了")
	enhancer_cnt = len(sentences)
	print(f"エンハンサーの個数 {enhancer_cnt}")

	print(f"{cell_line} の プロモーター...")
	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
	promoter_reverse_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_r_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")
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
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, fasta_line))
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, reverse_fasta_line))
			promoter_id += 1

	promoter_fasta_file.close()
	promoter_reverse_fasta_file.close()
	print(f"{cell_line} の プロモーター 終了")
	print(f"プロモーターの個数 {len(sentences) - enhancer_cnt}")

	# _____________________________________
	corpus = []
	for (tag, sentence) in zip(tags, sentences):
		corpus.append(TaggedDocument(sentence, [tag])) # doc2vec前のsentenceへのtagつけ

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
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, fasta_line))
				tags.append(tag) # reverse complement用に２回
				sentences.append(make_kmer_list(args.k, args.stride, reverse_fasta_line))
			enhancer_id += 1

	enhancer_fasta_file.close()
	enhancer_reverse_fasta_file.close()
	print(f"{cell_line} の エンハンサー 終了")
	print(f"エンハンサーの個数 {len(sentences)}")
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
	model.save(f"{args.my_data_folder_path}/model/{cell_line}_enhancer_{args.E_extended_left_length}_{args.E_extended_right_length}.model")


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
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, fasta_line))
				tags.append(tag)
				sentences.append(make_kmer_list(args.k, args.stride, reverse_fasta_line))
			promoter_id += 1

	promoter_fasta_file.close()
	promoter_reverse_fasta_file.close()
	print(f"{cell_line} の プロモーター 終了")
	print(f"プロモーターの個数 {len(sentences)}")

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
	model.save(f"{args.my_data_folder_path}/model/{cell_line}_promoter_{args.P_extended_left_length}_{args.P_extended_right_length}.model")