from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools
import tqdm

import os
import argparse

import gzip

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pybedtools

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import data_download


def make_extended_bedfile(args, cell_line):
	# エンハンサー，プロモーターの延長後の領域を記録したbedファイルを作成
	# paragraph tag もこのbedファイルに付与しておく

	data_download.download_chrome_sizes(args)
	chrom_sizes_df = pd.read_csv(os.path.join(args.my_data_folder_path, "reference_genome", "hg19_chrom_sizes.csv"))

	for region_type in ["enhancer", "promoter"]:
		origin_bed_df = pd.read_csv(f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed.csv") # original data
		extended_bed_path = f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed"

		print(f"{region_type} make extended bed")
		with open(extended_bed_path, "w") as fout:
			# for index, row_data in tqdm.tqdm(origin_bed_df.iterrows()):
			for index, row_data in origin_bed_df.iterrows():
				start, end = -1, -1 # 初期化
				chrom = row_data["chrom"]
				name = row_data["name_origin"]

				if region_type == "enhancer":
					start = row_data["start_origin"] - args.E_extended_left_length
					end = row_data["end_origin"] + args.E_extended_right_length
				elif region_type == "promoter":
					start = row_data["start_origin"] - args.P_extended_left_length
					end = row_data["end_origin"] + args.P_extended_right_length

				chrom_max_size = chrom_sizes_df[chrom_sizes_df["chrom"] == chrom]["size"].to_list()[0]
				if chrom_max_size < end:
					continue

				# paragraph_tag = region_type + "_" + str(index) # enhancer_0 など
				# fout.write(f"{chrom}\t{start}\t{end}\t{paragraph_tag}~{name}\n")
				fout.write(f"{chrom}\t{start}\t{end}\t{name}\n")


def make_extended_fastafile(args, cell_line):
	# エンハンサー，プロモーターの延長後の領域を記録したfastaファイルを作成
	# pybedtools で bed -> fasta
	for region_type in ["enhancer", "promoter"]:
		reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

		if os.path.exists(reference_genome_path) == False:
			# reference genome の解凍　解凍しなくてもgetfastaする方法を模索中
			print("unzip reference genome...")
			with gzip.open(reference_genome_path + ".gz", "rt") as fin, open(reference_genome_path, "w") as fout:
				fout.write(fin.read())
			print("unzipped!!")

		bed_path = f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed"
		output_fasta_path = f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s.fa"
		bed = pybedtools.BedTool(bed_path)
		print(f"{region_type} bed to fasta...")
		seq = bed.sequence(fi=reference_genome_path, nameOnly=True) # ここで切り出しているらしい
		with open(output_fasta_path, "w") as fout:
			fout.write(open(seq.seqfn).read())


def edit_fastafile(args, cell_line):
	# 配列内の欠損データを削除したり，配列内の文字を全て小文字にしたりします．
	# また，配列のcomplementも追加します．
	for region_type in ["enhancer", "promoter"]:

		input_fasta_path = f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s.fa"
		output_fasta_path = f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s_editted.fa"

		print(f"{region_type} edit fasta...")
		with open(input_fasta_path, "r") as fin, open(output_fasta_path, "w") as fout:
			for record in SeqIO.parse(fin, "fasta"):
				seq = str(record.seq).lower()
				if seq.count("n") > 0: # 欠損配列を除外
					continue
				complement_seq = str(record.seq.complement()).lower()

				fout.write(">" + str(record.id) + "\n")
				fout.write(seq + "\n")
				fout.write(">" + str(record.id) + " complement" + "\n") # complement 配列
				fout.write(complement_seq + "\n")


def create_region_sequence_and_table(args, cell_line):
	make_extended_bedfile(args, cell_line)
	make_extended_fastafile(args, cell_line)
	edit_fastafile(args, cell_line)