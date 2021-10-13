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


def create_extended_EnhPrm(args, cell_line):
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# input bed
	extended_enhancer_bed_path = f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.bed"
	extended_promoter_bed_path = f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.bed"

	# output fasta
	extended_enhancer_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
	extended_promoter_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

	extended_enhancer_r_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_r_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
	extended_promoter_r_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_r_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

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
	# reverse complement 作成
	seqs = ""
	reverse_seqs = "" # reverse complement
	with open(extended_enhancer_fasta_path, "r") as fout:
		seqs = fout.read()
	seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
	reverse_seqs = seqs.replace("a", "¥").replace("t", "a").replace("¥", "t").replace("c", "¥").replace("g", "c").replace("¥", "g").replace("ghr", "chr")
	with open(extended_enhancer_fasta_path, "w") as fout:
		fout.write(seqs)
	with open(extended_enhancer_r_fasta_path, "w") as fout:
		fout.write(reverse_seqs)
	
	with open(extended_promoter_fasta_path, "r") as fout:
		seqs = fout.read()
	seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
	reverse_seqs = seqs.replace("a", "¥").replace("t", "a").replace("¥", "t").replace("c", "¥").replace("g", "c").replace("¥", "g").replace("ghr", "chr")
	with open(extended_promoter_fasta_path, "w") as fout:
		fout.write(seqs)
	with open(extended_promoter_r_fasta_path, "w") as fout:
		fout.write(reverse_seqs)


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

	enhancer_id = 0
	names = [] # chr1:900000-9100000 など
	region_tags = [] # ENHANCER_0 などの tag を入れていく
	chrs = [] # chr1 などを入れていく
	starts = []	# start pos を入れていく
	ends = [] # end pos を入れていく
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = enhancer_fasta_file.readlines()

	for fasta_line in fasta_lines: # fasta file を 1行ずつ確認
		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行
			region_tag = "ENHANCER_" + str(enhancer_id)
			region_tags.append(region_tag)
			name = fasta_line[1:].replace("\n", "")
			chr, range_txt = name.split(":")[0], name.split(":")[1]
			start_pos, end_pos = range_txt.split("-")[0], range_txt.split("-")[1]
			names.append(name)
			chrs.append(chr)
			starts.append(start_pos)
			ends.append(end_pos)
		else: # 実際の塩基配列 nの個数を調べる
			n_cnt = fasta_line.count("n")
			n_cnts.append(n_cnt)
			enhancer_id += 1


	df = pd.DataFrame({
		"name":names,
		"tag":region_tags,
		"chr":chrs,
		"start":starts,
		"end":ends,
		"n_cnt":n_cnts,
	})
	df.to_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.csv")

	enhancer_fasta_file.close()
	print(f"エンハンサーの領域情報をcsvファイルにて保存完了")


	print(f"プロモーター...")
	promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")

	promoter_id = 0
	names = [] # chr1:900000-9100000 など
	region_tags = [] # PROMOTER_0 などの id を入れていく
	chrs = [] # chr1 などを入れていく
	starts = []	# start pos を入れていく
	ends = [] # end pos を入れていく
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = promoter_fasta_file.readlines()

	for fasta_line in fasta_lines:

		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行
			region_tag = "PROMOTER_" + str(promoter_id)
			region_tags.append(region_tag)

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
			promoter_id += 1



	df = pd.DataFrame({
		"name":names,
		"tag":region_tags,
		"chr":chrs,
		"start":starts,
		"end":ends,
		"n_cnt":n_cnts,
	})
	df.to_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv")

	promoter_fasta_file.close()
	print(f"プロモーターの領域情報をcsvファイルにて保存完了")

def create_region_bedfile_and_table(args, cell_line):
    create_extended_EnhPrm(args, cell_line)
    make_extended_region_table(args, cell_line)