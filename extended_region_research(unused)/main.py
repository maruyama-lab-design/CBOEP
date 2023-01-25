from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse

import datetime

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 自作関数
from make_directory import make_directory
import data_download
from data_processing import create_region_sequence_and_table
import my_doc2vec
import train_classifier
from make_args_logfile import make_args_logfile

def my_project(args, cell_line):

	print("実験開始")
	# args.output = f"{cell_line},el={str(args.E_extended_left_length)},er={str(args.E_extended_right_length)},pl={str(args.P_extended_left_length)},pr={str(args.P_extended_right_length)},share_doc2vec={str(args.share_doc2vec)},d={args.embedding_vector_dimention}kmer={args.way_of_kmer},N={args.sentence_cnt}"
	args.output = f"{cell_line},d={args.embedding_vector_dimention},way_of_kmer={args.way_of_kmer},k={args.k},s={args.stride},N={args.sentence_cnt},kmin={args.k_min},kmax={args.k_max},way_of_cv={args.way_of_cv},clf={args.classifier},trees={args.gbrt_tree_cnt},train_data={args.research_name}"
	print(f"output = {args.output}")

	# 必要なディレクトリの作成
	if args.make_directory:
		make_directory(args)

	# リファレンスゲノムのダウンロード
	if args.download_reference_genome:
		# data_download.download_reference_genome(args)
		args.download_reference_genome = False # 一回のみ

	# エンハンサープロモーターのダウンロード
	# data_download.download_enhancer_and_promoter(args, cell_line)

	if os.path.exists(args.output): # 存在してたらスキップ
		print(args.output + " スキップ")
		return 
	
	# bedの作成&fastaの作成&tableの作成
	# create_region_sequence_and_table(args, cell_line)

	# doc2vec (stage1)
	if args.stage2_only == False:
		args.stage1_start_time = datetime.datetime.now()
		my_doc2vec.make_paragraph_vector_from_enhancer_and_promoter_using_iterator(args, cell_line)
		args.stage1_end_time = datetime.datetime.now()

	# 分類期学習 (stage2)
	args.stage2_start_time = datetime.datetime.now()
	train_classifier.train(args, cell_line)
	args.stage2_end_time = datetime.datetime.now()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--genome_browser_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest")
	parser.add_argument("--research_name", type=str, choices=['ep2vec', 'TargetFinder', 'my'], help="どこ由来の学習データを使うか", default="my")
	parser.add_argument("-my_data_folder_path", help="自分のデータフォルダパス", default="/Users/ylwrvr/卒論/Koga_code/data")
	parser.add_argument("--make_directory", action="store_true", help="実験に必要なディレクトリ構成を作る")
	parser.add_argument("--download_reference_genome", action="store_true", help="リファレンスゲノムを外部からダウンロードするか")
	parser.add_argument("--share_doc2vec", action="store_true", help="エンハンサーとプロモーターを一つのdoc2vecに共存させるか")
	parser.add_argument("--cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["K562"])
	parser.add_argument("-el", "--E_extended_left_length", type=int, default=0, help="エンハンサーの上流をどれだけ伸ばすか")
	parser.add_argument("-er", "--E_extended_right_length", type=int, default=0, help="エンハンサーの下流をどれだけ伸ばすか")
	parser.add_argument("-pl", "--P_extended_left_length", type=int, default=0, help="プロモーターの上流をどれだけ伸ばすか")
	parser.add_argument("-pr", "--P_extended_right_length", type=int, default=0, help="プロモーターの下流をどれだけ伸ばすか")
	parser.add_argument("--embedding_vector_dimention", type=int, default=100, help="paragraph vector の次元")
	parser.add_argument('--way_of_kmer', type=str, choices=['normal', 'random'], default="normal", help='k-merの切り方 固定長かランダム長か')
	parser.add_argument("--k", type=int, default=6, help="固定長のk-merの場合のk")
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--sentence_cnt", type=int, default=5, help="ランダム長のk-merの場合,一本のseqから生成されるのsentence個数")
	parser.add_argument("--k_min", type=int, default=3, help="ランダム長のk-merの場合のk_min")
	parser.add_argument("--k_max", type=int, default=6, help="ランダム長のk-merの場合のk_max")
	parser.add_argument("--classifier", type=str, choices=["GBRT"], default="GBRT", help="分類器に何を使うか")
	parser.add_argument("--way_of_cv", type=str, choices=["random", "split"], default="split", help="ランダムcross-valか，染色体番号ごとか")
	parser.add_argument("--gbrt_tree_cnt", type=int, default=4000, help="GBRTの木の数")
	parser.add_argument("--stage1_start_time", type=str, help="doc2vec開始時間")
	parser.add_argument("--stage1_end_time", type=str, help="doc2vec終了時間")
	parser.add_argument("--stage2_start_time", type=str, help="分類期学習開始時間")
	parser.add_argument("--stage2_end_time", type=str, help="分類期学習終了時間")
	parser.add_argument("--output", type=str, help="output名")
	parser.add_argument("--stage2_only", action="store_true", help="分類器学習のみ")
	args = parser.parse_args()

	if args.way_of_kmer == "normal":
		args.sentence_cnt = 1
		args.k_min = -1
		args.k_max = -1
	elif args.way_of_kmer == "random":
		args.k = -1
		args.stride = -1

	
	for cell_line in args.cell_line_list:
		my_project(args, cell_line)
		make_args_logfile(args)

