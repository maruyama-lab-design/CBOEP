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
from make_dirictory import make_directory
import data_download
from data_processing import data_processing
import my_doc2vec
import train_classifier


### def func(args):



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
	parser.add_argument("--cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["K562"])
	parser.add_argument("--neighbor_length", default=5000)
	parser.add_argument("-el", "--E_extended_left_length", type=int, default=0, help="エンハンサーの上流をどれだけ伸ばすか")
	parser.add_argument("-er", "--E_extended_right_length", type=int, default=0, help="エンハンサーの下流をどれだけ伸ばすか")
	parser.add_argument("-pl", "--P_extended_left_length", type=int, default=0, help="プロモーターの上流をどれだけ伸ばすか")
	parser.add_argument("-pr", "--P_extended_right_length", type=int, default=0, help="プロモーターの下流をどれだけ伸ばすか")
	parser.add_argument("--embedding_vector_dimention", type=int, default=100, help="paragraph vector の次元")
	parser.add_argument("--k", type=int, default=6, help="doc2vec前処理のk-merのため")
	parser.add_argument("--stride", type=int, default=1, help="doc2vec前処理のk-merのため")
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("--make_directory", action="store_true", help="実験に必要なディレクトリ構成を作る")
	parser.add_argument("--download_reference_genome", action="store_true", help="リファレンスゲノムを外部からダウンロードするか")
	parser.add_argument("--share_doc2vec", action="store_true", help="エンハンサーとプロモーターを一つのdoc2vecに共存させるか")
	args = parser.parse_args()

	### Make the remaining part as a function like def func(args), and put it above. 

	### save the contents of args to a log file.

	# 必要なディレクトリの作成
	if args.make_directory:
		make_directory(args)

	# リファレンスゲノムのダウンロード
	if args.download_reference_genome:
		data_download.download_reference_genome(args)


	for cell_line in args.cell_line_list: # 細胞株毎のループ

		### It would be better to make this part as a function. 

		# エンハンサープロモーターのダウンロード
		data_download.download_enhancer_and_promoter(args, cell_line)

		pl_list = [0] # 意味が分かる変数名にしてほしい．
		pr_list = [0]
		el_list = [0]
		er_list = [0]
		for (el, er, pl, pr) in list(itertools.product(el_list, er_list, pl_list, pr_list)): # 組み合わせ全列挙
			args.E_extended_left_length = el
			args.E_extended_right_length = er
			args.P_extended_left_length = pl
			args.P_extended_right_length = pr

			output = f"{args.my_data_folder_path}/result/{cell_line},el={str(args.E_extended_left_length)},er={str(args.E_extended_right_length)},pl={str(args.P_extended_left_length)},pr={str(args.P_extended_right_length)},share_doc2vec={str(args.share_doc2vec)}.csv"
			if os.path.exists(output): # 存在してたらスキップ
				print(f"{cell_line},el={str(args.E_extended_left_length)},er={str(args.E_extended_right_length)},pl={str(args.P_extended_left_length)},pr={str(args.P_extended_right_length)},share_doc2vec={str(args.share_doc2vec)} スキップ")
				continue
			
			data_processing(args, cell_line) ### どのようなプロセスかが分かる関数名がいいです．

			# doc2vec
			if args.share_doc2vec:
				my_doc2vec.make_paragraph_vector_from_enhancer_and_promoter(args, cell_line)
			else:
				my_doc2vec.make_paragraph_vector_from_enhancer_only(args, cell_line)
				my_doc2vec.make_paragraph_vector_from_promoter_only(args, cell_line)

			train_classifier.train(args, cell_line)

