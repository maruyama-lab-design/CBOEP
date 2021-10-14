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
from make_dirictory import make_directory
import data_download
from data_processing import create_region_sequence_and_table
import my_doc2vec
import train_classifier
from make_args_logfile import make_args_logfile

# from data_processing import create_promoter_bedfile_divided_from_tss


def my_project(args):

	# 必要なディレクトリの作成
	if args.make_directory:
		make_directory(args)

	# リファレンスゲノムのダウンロード
	if args.download_reference_genome:
		data_download.download_reference_genome(args)

	promoter_left_length_list = [0] # 意味が分かる変数名にしてほしい．
	promoter_right_length_list = [0]
	enhancer_left_length_list = [0]
	enhancer_right_length_list = [0]

	for cell_line in args.cell_line_list: # 細胞株毎のループ

		### It would be better to make this part as a function. 

		# エンハンサープロモーターのダウンロード
		data_download.download_enhancer_and_promoter(args, cell_line)

		for (enhancer_left_length, enhancer_right_length, promoter_left_length, promoter_right_length) in list(itertools.product(enhancer_left_length_list, enhancer_right_length_list, promoter_left_length_list, promoter_right_length_list)): # 組み合わせ全列挙

			args.E_extended_left_length = enhancer_left_length
			args.E_extended_right_length = enhancer_right_length
			args.P_extended_left_length = promoter_left_length
			args.P_extended_right_length = promoter_right_length

			args.output = f"{cell_line},el={str(args.E_extended_left_length)},er={str(args.E_extended_right_length)},pl={str(args.P_extended_left_length)},pr={str(args.P_extended_right_length)},share_doc2vec={str(args.share_doc2vec)},kmer={args.way_of_kmer}"
			if os.path.exists(args.output): # 存在してたらスキップ
				print(args.output + " スキップ")
				continue
			
			create_region_sequence_and_table(args, cell_line) ### どのようなプロセスかが分かる関数名がいいです．

			# doc2vec
			args.stage1_start_time = datetime.datetime.now()
			if args.share_doc2vec:
				my_doc2vec.make_paragraph_vector_from_enhancer_and_promoter(args, cell_line)
			else:
				my_doc2vec.make_paragraph_vector_from_enhancer_only(args, cell_line)
				my_doc2vec.make_paragraph_vector_from_promoter_only(args, cell_line)
			args.stage1_end_time = datetime.datetime.now()

			# 分類期学習
			args.stage2_start_time = datetime.datetime.now()
			train_classifier.train(args, cell_line)
			args.stage2_end_time = datetime.datetime.now()

			# make logfile
			make_args_logfile(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
	parser.add_argument("--cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["K562"])
	parser.add_argument("-el", "--E_extended_left_length", type=int, default=0, help="エンハンサーの上流をどれだけ伸ばすか")
	parser.add_argument("-er", "--E_extended_right_length", type=int, default=0, help="エンハンサーの下流をどれだけ伸ばすか")
	parser.add_argument("-pl", "--P_extended_left_length", type=int, default=0, help="プロモーターの上流をどれだけ伸ばすか")
	parser.add_argument("-pr", "--P_extended_right_length", type=int, default=0, help="プロモーターの下流をどれだけ伸ばすか")
	parser.add_argument("--make_promoter_from_tss", action="store_true", help="プロモーターをtss基準に生成するか")
	parser.add_argument("--embedding_vector_dimention", type=int, default=100, help="paragraph vector の次元")
	parser.add_argument('--way_of_kmer', type=str, choices=['normal', 'random'], help='k-merの切り方')
	parser.add_argument("--k", type=int, default=6, help="固定長のk-merの場合のk")
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--sentence_cnt", type=int, default=10, help="ランダム長のk-merの場合,一本のseqから生成されるのsentence個数")
	parser.add_argument("--k_min", type=int, default=3, help="ランダム長のk-merの場合のk_min")
	parser.add_argument("--k_max", type=int, default=6, help="ランダム長のk-merの場合のk_max")
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("--make_directory", action="store_true", help="実験に必要なディレクトリ構成を作る")
	parser.add_argument("--download_reference_genome", action="store_true", help="リファレンスゲノムを外部からダウンロードするか")
	parser.add_argument("--share_doc2vec", action="store_true", help="エンハンサーとプロモーターを一つのdoc2vecに共存させるか")
	parser.add_argument("--stage1_start_time", type=str, help="doc2vec開始時間")
	parser.add_argument("--stage1_end_time", type=str, help="doc2vec終了時間")
	parser.add_argument("--stage2_start_time", type=str, help="分類期学習開始時間")
	parser.add_argument("--stage2_end_time", type=str, help="分類期学習終了時間")
	parser.add_argument("--output")
	args = parser.parse_args()

	args.way_of_kmer = "random"

	### Make the remaining part as a function like def func(args), and put it above. 
	my_project(args)

	### save the contents of args to a log file.



