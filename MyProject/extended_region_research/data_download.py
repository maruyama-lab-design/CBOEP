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
		bed_df = pd.read_csv(url,sep="\t")
		bed_df.columns = ["chrom", "start_origin", "end_origin", "name_origin"]
		bed_path = os.path.join(args.my_data_folder_path, "bed", region_type, f"{cell_line}_{region_type}s.bed.csv")
		bed_df.to_csv(bed_path, index=False)

	
def download_reference_genome(args):
	# reference genome
	print("reference genome downloading...")
	url = f"{args.reference_genome_url}"
	output = os.path.join(args.my_data_folder_path, "reference_genome", "hg19.fa.gz")
	urllib.request.urlretrieve(url, output)

def download_training_data(args, cell_line):
	print("training data downloading...")
	url = f"https://raw.githubusercontent.com/wanwenzeng/ep2vec/master/{cell_line}train.csv"
	df = pd.read_csv(url, usecols=["bin", "enhancer_name", "promoter_name", "label"])
	df.to_csv(f"{args.my_data_folder_path}/train/{cell_line}_train.csv", index=False)