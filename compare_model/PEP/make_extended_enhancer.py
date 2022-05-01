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

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pybedtools

# get_motifVec -> motif vector を return
# get_wordVec -> word vector を return

def make_extended_enhancer_bed(args):
	# enhancerのbedfileを読み込んでstartとendを変える．
	# この際，nameは変えないよう注意する．

	origin_bed_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sequence", args.cell_line, "enhancers.bed"), sep='\t', comment='t', header=None)
	origin_bed_df.columns = ["chrom", "start", "end", "name"]

	origin_bed_df["start"] -= args.flanking_region_length
	origin_bed_df["end"] += args.flanking_region_length

	assert origin_bed_df.eval("start < end").all

	origin_bed_df.to_csv(os.path.join(os.path.dirname(__file__), "extended_seq", args.cell_line, "enhancers.bed"), sep="\t", header=False, index=False)


def make_extended_enhancer_fasta(args):
	referenceGenome_path = os.path.join(os.path.dirname(__file__), "..", "ep2vec_preprocess", "hg19.fa")


	bed_path = os.path.join(os.path.dirname(__file__), "extended_seq", args.cell_line, "enhancers.bed")

	output_dir = os.path.join(os.path.dirname(__file__), "extended_seq", args.cell_line)
	output_path = os.path.join(output_dir, "enhancers.fa")
	os.system(f"mkdir -p {output_dir}")

	print(f"making {args.cell_line} extended enhancer fasta...")
	bed = pybedtools.BedTool(bed_path)
	seq = bed.sequence(fi=referenceGenome_path, nameOnly=True) # ここで切り出しているらしい
	with open(output_path, "w") as fout:
		fout.write(open(seq.seqfn).read())
	print("completed")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinder実行")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--ratio", type=int, help="正例に対し何倍の負例があるか", default="1")
	parser.add_argument("--wordVec_dimention", type=int, help="enhancer or promoter のwordVecの次元", default=300)
	parser.add_argument("--flanking_region_length", type=int, help="enhancerの両端をどれだけ伸ばすか", default=4000)
	parser.add_argument("--mode", default="integrated")
	parser.add_argument("--output_dir", default="")
	args = parser.parse_args()

	# make_extended_enhancer_bed(args)
	make_extended_enhancer_fasta(args)