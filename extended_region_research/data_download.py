from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import urllib.request
import gzip
from Bio import SeqIO

import os
import argparse


# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def download_enhancer_and_promoter(args, cell_line):
	# enhancer
	for region_type in ["enhancer", "promoter"]:
		print(f"{region_type}　downloading...")
		url = os.path.join(args.targetfinder_data_root_url, cell_line, "output-ep", f"{region_type}s.bed")
		bed_df = pd.read_csv(url,sep="\t",names=["chrom", "start_origin", "end_origin", "name_origin"])
		# bed_df.columns = ["chrom", "start_origin", "end_origin", "name_origin"]
		bed_path = os.path.join(args.my_data_folder_path, "bed", region_type, f"{cell_line}_{region_type}s.bed.csv")
		bed_df.to_csv(bed_path, index=False)

	
def download_reference_genome(args):
	# reference genome
	print("reference genome downloading...")
	url = os.path.join(f"{args.genome_browser_url}", "hg19.fa.gz")
	output = os.path.join(args.my_data_folder_path, "reference_genome", "hg19.fa.gz")
	urllib.request.urlretrieve(url, output)

def download_chrome_sizes(args):
	print("chrome_sizes downloading...")
	url = os.path.join(f"{args.genome_browser_url}", "hg19.chrom.sizes")
	df = pd.read_csv(url, sep="\t", names=["chrom", "size"])
	# df.columns = ["chrom", "size"]
	output = os.path.join(args.my_data_folder_path, "reference_genome", "hg19_chrom_sizes.csv")
	df.to_csv(output, index=False)

def download_training_data(args, cell_line):
	targetfinder_output_root = "https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder"
	ep2vec_root = "https://github.com/wanwenzeng/ep2vec/raw/master"
	
	print("training data downloading...")

	# training data の url (TargetFinder)
	targetfinder_url = os.path.join(targetfinder_output_root, cell_line, "output-ep", "training.csv.gz")
	targetfinder_train_df = pd.read_csv(targetfinder_url,compression='gzip',error_bad_lines=False)

	# training data の url (ep2vec)
	ep2vec_url = os.path.join(ep2vec_root, f"{cell_line}train.csv")
	ep2vec_train_df = pd.read_csv(ep2vec_url)

	# 保存する名前 data下に置く
	targetfinder_training_data_filename = os.path.join(args.my_data_folder_path, "train", "TargetFinder", f"{cell_line}_train.csv")
	targetfinder_train_df.to_csv(targetfinder_training_data_filename, index=False)

	ep2vec_training_data_filename = os.path.join(args.my_data_folder_path, "train", "ep2vec", f"{cell_line}_train.csv")
	ep2vec_train_df.to_csv(ep2vec_training_data_filename, index=False)