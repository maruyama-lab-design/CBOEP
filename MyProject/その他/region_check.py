from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import os
import argparse

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier

def check_enhancer_length(args, cell_line):
	bin = 500
	with open(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed") as f:
		lines = f.readlines()
		length = [0] * len(lines)
		for i, line in enumerate(lines):
			start = int(line.split("\t")[1])
			end = int(line.split("\t")[2])
			length[i] = (end - start) // bin
	length_cnt = Counter(length)
	length_cnt = length_cnt.items()
	length_cnt = sorted(length_cnt) 
	x, y = zip(*length_cnt)
	print(x[-20:])

	figure = plt.figure()
	plt.title(f"{cell_line} enhancer")
	plt.xlabel(f"enhancer length(×{bin})")
	plt.ylabel(f"cnt")
	plt.bar(x, y)
	# plt.show()

	figure.savefig(f"/Users/ylwrvr/卒論/Koga_code/figure/{cell_line}_enhancer_count_bin={bin}.jpg", dpi=300)


def check_promoter_length(args, cell_line):
	bin = 1000
	with open(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed") as f:
		lines = f.readlines()
		length = [0] * len(lines)
		for i, line in enumerate(lines):
			start = int(line.split("\t")[1])
			end = int(line.split("\t")[2])
			length[i] = (end - start) // bin
	length_cnt = Counter(length)
	length_cnt = length_cnt.items()
	length_cnt = sorted(length_cnt) 
	x, y = zip(*length_cnt)
	print(x[-20:])

	figure = plt.figure()
	plt.title(f"{cell_line} promoter")
	plt.xlabel(f"promoter length(×{bin})")
	plt.ylabel(f"cnt")
	plt.bar(x, y)
	# plt.show()

	figure.savefig(f"/Users/ylwrvr/卒論/Koga_code/figure/{cell_line}_promoter_count_bin={bin}.jpg", dpi=300)


def check_distance_length(args, cell_line):
	bin = 10000
	df = pd.read_csv(f"{args.my_data_folder_path}/train/{cell_line}_train.csv", usecols=["label", "enhancer_distance_to_promoter"])
	enhancer_distance_to_promoter = df["enhancer_distance_to_promoter"].tolist()
	enhancer_distance_to_promoter = list(map(lambda x: x // bin, enhancer_distance_to_promoter))
	distance_cnt = Counter(enhancer_distance_to_promoter)
	distance_cnt = distance_cnt.items()
	distance_cnt = sorted(distance_cnt)
	x, y = zip(*distance_cnt)

	figure = plt.figure()
	plt.title(f"{cell_line} enhancer-promoter distance")
	plt.xlabel(f"distance (×{bin})")
	plt.ylabel(f"cnt")
	plt.bar(x, y)
	figure.savefig(f"/Users/ylwrvr/卒論/Koga_code/figure/{cell_line}_enhancer_distance_to_promoter_count_bin={bin}.jpg", dpi=300)

	label_groupby = df.groupby("label")
	for label, group in label_groupby:
		enhancer_distance_to_promoter = group["enhancer_distance_to_promoter"].tolist()
		enhancer_distance_to_promoter = list(map(lambda x: x // bin, enhancer_distance_to_promoter))
		distance_cnt = Counter(enhancer_distance_to_promoter)
		distance_cnt = distance_cnt.items()
		distance_cnt = sorted(distance_cnt)
		x, y = zip(*distance_cnt)

		figure = plt.figure()
		plt.title(f"{cell_line} enhancer-promoter distance (label={label})")
		plt.xlabel(f"distance (×{bin})")
		plt.ylabel(f"cnt")
		plt.bar(x, y)
		# plt.show()
		figure.savefig(f"/Users/ylwrvr/卒論/Koga_code/figure/{cell_line}_enhancer_distance_to_promoter_count_bin={bin}_label={label}.jpg", dpi=300)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
	parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", default=5000)
	parser.add_argument("-E_extended_left_length", type=int, default=0)
	parser.add_argument("-E_extended_right_length", type=int, default=0)
	parser.add_argument("-P_extended_left_length", type=int, default=0)
	parser.add_argument("-P_extended_right_length", type=int, default=0)
	parser.add_argument("-embedding_vector_dimention", type=int, default=100)
	parser.add_argument("-k", type=int, default=6)
	parser.add_argument("-stride", type=int, default=1)
	args = parser.parse_args()

	
	for cell_line in args.cell_line_list:
		# check_enhancer_length(args, cell_line)
		# check_promoter_length(args, cell_line)
		check_distance_length(args, cell_line)


