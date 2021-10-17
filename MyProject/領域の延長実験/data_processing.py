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


def create_region_sequence(args, cell_line):
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# input bed
	extended_enhancer_bed_path = f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.bed"
	extended_promoter_bed_path = f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.bed"

	# output fasta (forward)
	extended_enhancer_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
	extended_promoter_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

	# output fasta (reverse)
	extended_enhancer_r_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_r_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
	extended_promoter_r_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_r_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

	if not os.path.exists(extended_enhancer_bed_path):
		print("与えられたエンハンサーのbedfileがありません")
		print("オリジナルのエンハンサーのbedfileから作成します...")
		text = ""
		with open(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed", "r") as origin_bed:
			lines = origin_bed.readlines()
			for line in lines: # 一行ずつbedfileを読み込む
				line = line.split("\t")
				chr, start_pos, end_pos, name = line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")
				start_pos -= args.E_extended_left_length # 上流を伸ばす
				end_pos += args.E_extended_right_length # 下流を伸ばす
				text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n" # bedfile形式に書き込む
		with open(extended_enhancer_bed_path, "w") as extended_bed:
			extended_bed.write(text)

	if not os.path.exists(extended_promoter_bed_path):
		print("与えられたプロモーターのbedfileがありません")
		print("オリジナルのプロモーターのbedfileから作成します...")
		text = ""
		with open(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed", "r") as origin_bed:
			lines = origin_bed.readlines()
			for line in lines: # 一行ずつ読み込み
				line = line.split("\t")
				chr, start_pos, end_pos, name = line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")
				start_pos -= args.P_extended_left_length # 上流を伸ばす
				end_pos += args.P_extended_right_length # 下流を伸ばす
				text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n" # bedfile形式に書き込む
		with open(extended_promoter_bed_path, "w") as extended_bed:
			extended_bed.write(text)


	# bedtools で hg19 を bed 切り出し → fasta に保存
	print("bedfileからfastafileを作ります")
	os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_enhancer_bed_path +" -fo "+ extended_enhancer_fasta_path -name)
	os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_promoter_bed_path +" -fo "+ extended_promoter_fasta_path -name)

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


def make_region_table(args, cell_line):
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
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = enhancer_fasta_file.readlines()

	for fasta_line in fasta_lines: # fasta file を 1行ずつ確認
		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行
			region_tag = "ENHANCER_" + str(enhancer_id)
			region_tags.append(region_tag)
			name = fasta_line[1:].replace("\n", "") # "chr1:17000-18000"
			names.append(name)
		else: # 実際の塩基配列 nの個数を調べる
			n_cnt = fasta_line.count("n")
			n_cnts.append(n_cnt)
			enhancer_id += 1

	# pandas を使って csv に保存
	df = pd.DataFrame({
		"name":names,
		"tag":region_tags,
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
	n_cnts = [] # sequence 内の "n" の個数を入れていく

	fasta_lines = promoter_fasta_file.readlines()

	for fasta_line in fasta_lines:

		if fasta_line[0] == ">": # ">chr1:17000-18000" のような行
			region_tag = "PROMOTER_" + str(promoter_id)
			region_tags.append(region_tag)
			name = fasta_line[1:].replace("\n", "")
			names.append(name)
		else:
			n_cnt = fasta_line.count("n") # 配列内の"n"の個数を数える
			n_cnts.append(n_cnt)
			promoter_id += 1


	# pandas
	df = pd.DataFrame({
		"name":names,
		"tag":region_tags,
		"n_cnt":n_cnts,
	})
	df.to_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv")

	promoter_fasta_file.close()
	print(f"プロモーターの領域情報をcsvファイルにて保存完了")


def create_promoter_bedfile_divided_from_tss(args, cell_line):
	print("tssデータをダウンロード...")
	os.system(f"wget {args.targetfinder_data_root_url}/{cell_line}/output-ep/tss.bed -O {args.my_data_folder_path}/bed/tss/{cell_line}_tss.bed")
	print("ダウンロードしたtssのbedfileを編集")
	tss_bed_table = pd.read_csv(
		f"{args.my_data_folder_path}/bed/tss/{cell_line}_tss.bed",
		sep="\t",
		header = None,
		usecols=[0, 1]
	)
	tss_bed_table.columns = ["chr", "pos"]
	tss_bed_table["pos"] = tss_bed_table["pos"].astype(int)

	promoter_bed_table = pd.read_csv(
		f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed",
		sep="\t",
		header = None,
	)
	promoter_bed_table.columns = ['chr', 'start', 'end', 'name']
	promoter_bed_table["start"] = promoter_bed_table["start"].astype(int)
	promoter_bed_table["end"] = promoter_bed_table["end"].astype(int)

	drop_indexes = []
	new_tss_bed_txt = ""
	for index, tss_data in tss_bed_table.iterrows():
		now_chr = tss_data["chr"]
		now_tss_pos = tss_data["pos"]
		if len(promoter_bed_table[(promoter_bed_table["chr"] == now_chr) & (promoter_bed_table["start"] < now_tss_pos) & (now_tss_pos < promoter_bed_table["end"])]) == 0:
			drop_indexes.append(index)
		elif len(promoter_bed_table[(promoter_bed_table["chr"] == now_chr) & (promoter_bed_table["start"] < now_tss_pos) & (now_tss_pos < promoter_bed_table["end"])]) == 1:
			promoter_name = promoter_bed_table[(promoter_bed_table["chr"] == now_chr) & (promoter_bed_table["start"] < now_tss_pos) & (now_tss_pos < promoter_bed_table["end"])]["name"]
			new_tss_bed_txt += str(now_chr) + "\t" + str(now_tss_pos) + "\t" + str(now_tss_pos) + "\t" + str(promoter_name) + "\n"
		else:
			print("error")
			exit()


	tss_bed_table = tss_bed_table.drop(tss_bed_table.index[drop_indexes])
	with open("test.txt", "w") as f:
		f.write(new_tss_bed_txt)

	print(len(tss_bed_table))
	print(len(promoter_bed_table))


	# promoter_divided_from_tss_bedfile = open("")


def create_region_sequence_and_table(args, cell_line):
	create_region_sequence(args, cell_line)
	make_region_table(args, cell_line)