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

def data_download(args, cell_line):
	# enhancer
	print("エンハンサーをダウンロードします.")
	os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/enhancers.bed -O {args.my_data_folder_path}bed/enhancer/{cell_line}_enhancers.bed")

	# promoter
	print("プロモーターをダウンロードします.")
	os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/promoters.bed -O {args.my_data_folder_path}bed/promoter/{cell_line}_promoters.bed")

	# reference genome
	print("リファレンスゲノムをダウンロードします.")
	os.system(f"wget {args.reference_genome_url} -O {args.my_data_folder_path}reference_genome/hg19.fa.gz")

