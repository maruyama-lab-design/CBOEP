from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import requests
import gzip
from Bio import SeqIO

import os
import argparse
import urllib.request


# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def download_enhancer_and_promoter(args, cell_line):
	# enhancer
	print("エンハンサーをダウンロード...")
	url = f"{args.targetfinder_data_root_url}/{cell_line}/output-ep/enhancers.bed"
	enhancers_bed_df = pd.read_csv(url,sep="\t")
	enhancers_bed_df.columns = ["chrom", "start_origin", "end_origin", "name_origin"]
	# enhancers_bed = enhancers_bed.rename(index=lambda i: "ENHANCER_" + str(i))
	enhancers_bed_df.to_csv(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed.csv")

	# promoter
	print("プロモーターをダウンロード...")
	url = f"{args.targetfinder_data_root_url}/{cell_line}/output-ep/promoters.bed"
	promoters_bed_df = pd.read_csv(url,sep="\t")
	promoters_bed_df.columns = ["chrom", "start_origin", "end_origin", "name_origin"]
	# enhancers_bed = enhancers_bed.rename(index=lambda i: "ENHANCER_" + str(i))
	promoters_bed_df.to_csv(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed.csv")

	
def download_reference_genome(args):
	# reference genome
	print("リファレンスゲノムをダウンロード...")
	url = f"{args.reference_genome_url}"
	output_filename = f"{args.my_data_folder_path}/reference_genome/hg19.fa.gz"
	os.system(f"wget {url} -O {output_filename}")

	# 読み込む際は以下参照
	# with gzip.open(f"{output_filename}", "rt") as handle:
	# 	for record in SeqIO.parse(handle, "fasta"):
	# 		print(record.id) # -> chr1 chr2 ... 
	# 	foo = ""

	# os.system(f"wget {args.reference_genome_url} -O {args.my_data_folder_path}/reference_genome/hg19.fa.gz")
	# print("解凍...")
	# os.system(f"gunzip -f {args.my_data_folder_path}/reference_genome/hg19.fa.gz")
	
