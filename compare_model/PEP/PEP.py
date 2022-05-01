import pandas as pd
import numpy as np
import joblib
import os
import io
import subprocess
import tempfile
from glob import glob

import argparse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# get_motifVec -> motif vector を return
# get_wordVec -> word vector を return

def make_extended_enhancer_bed(args):
	# enhancerのbedfileを読み込んでstartとendを変える．
	# この際，nameは変えないよう注意する．
	pass

def make_extended_enhancer_fasta(args):
	pass


def make_promoter_fasta(args):
	pass

def get_motifVec(args):
    pass

def get_wordVec(args):
    pass




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinder実行")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--ratio", type=int, help="正例に対し何倍の負例があるか", default="1")
    parser.add_argument("--wordVec_dimention", type=int, help="enhancer or promoter のwordVecの次元", default=300)
	parser.add_argument("--flanking_region_length", type=int, help="enhancerの両端をどれだけ伸ばすか", default=4000)
    parser.add_argument("--mode", default="integrated")
	parser.add_argument("--output_dir", default="")
	args = parser.parse_args()