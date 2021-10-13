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

def download_enhancer_and_promoter(args, cell_line):
	# enhancer
	print("エンハンサーをダウンロード...")
	os.system(f"wget {args.targetfinder_data_root_url}/{cell_line}/output-ep/enhancers.bed -O {args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed")

	# promoter
	print("プロモーターをダウンロード...")
	os.system(f"wget {args.targetfinder_data_root_url}/{cell_line}/output-ep/promoters.bed -O {args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed")

	
def download_reference_genome(args):
	# reference genome
	print("リファレンスゲノムをダウンロード...")
	os.system(f"wget {args.reference_genome_url} -O {args.my_data_folder_path}/reference_genome/hg19.fa.gz")
	print("解凍...")
	os.system(f"gunzip -f {args.my_data_folder_path}/reference_genome/hg19.fa.gz")
	
