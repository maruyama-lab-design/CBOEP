# Here, features are added to the pair data.
# The addition of the window regions is also executed here.

#!/usr/bin/env python

import chromatics
import os
import numpy as np
import pandas as pd
import sys

from glob import glob
import math
import argparse

import json


def insert_window(df):
	cell = df.at[0, "enhancer_name"].split("|")[0]
	df["window_chrom"] = df["enhancer_chrom"]
	df["window_start"] = np.where(df["enhancer_end"] < df["promoter_end"], df["enhancer_end"], df["promoter_end"])
	df["window_end"] = np.where(df["enhancer_start"] > df["promoter_start"], df["enhancer_start"], df["promoter_start"])
	df["window_name"] = cell + "|" + df["window_chrom"] + ":" + df["window_start"].apply(str) + "-" +  df["window_end"].apply(str)
	assert (df["window_end"] >= df["window_start"]).all(), df[df["window_end"] <= df["window_start"]].head()
	return df


def preprocess_features(args):

	peaks_fn = 'peaks.bed.gz'
	methylation_fn = 'methylation.bed.gz'
	cage_fn = 'cage.bed.gz'
	generators = []

	# preprocess peaks
	peaks_dir = os.path.join(os.path.dirname(__file__), "input_features", args.cell, "peaks")
	if os.path.exists(peaks_dir):
		print(f"preprocess peaks...")
		assays = []
		for name, filename, source, accession in pd.read_csv(os.path.join(peaks_dir, "filenames.csv")).itertuples(index = False):
			columns = chromatics.narrowpeak_bed_columns if filename.endswith('narrowPeak') else chromatics.broadpeak_bed_columns
			assay_df = chromatics.read_bed(os.path.join(peaks_dir, f"{filename}.gz"), names = columns, usecols = chromatics.generic_bed_columns + ['signal_value'])
			assay_df['name'] = name
			assays.append(assay_df)
		peaks_df = pd.concat(assays, ignore_index = True)
		chromatics.write_bed(peaks_df, peaks_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, peaks_fn))

	# preprocess methylation
	methylation_dir = os.path.join(os.path.dirname(__file__), "input_features", args.cell, "methylation")
	if os.path.exists(methylation_dir):
		print(f"preprocess methylation...")
		assays = [chromatics.read_bed(_, names = chromatics.methylation_bed_columns, usecols = chromatics.generic_bed_columns + ['mapped_reads', 'percent_methylated']) for _ in glob(os.path.join(methylation_dir, f"*.bed.gz"))]
		methylation_df = pd.concat(assays, ignore_index = True).query('mapped_reads >= 10 and percent_methylated > 0')
		methylation_df['name'] = 'Methylation'
		del methylation_df['mapped_reads']
		chromatics.write_bed(methylation_df, methylation_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, methylation_fn))

	# preprocess cage
	cage_dir = os.path.join(os.path.dirname(__file__), "input_features", args.cell, "cage")
	if os.path.exists(cage_dir):
		print(f"preprocess cage...")
		cage_df = chromatics.read_bed(glob(os.path.join(cage_dir, f"*.bed.gz"))[0], names = chromatics.cage_bed_columns, usecols = chromatics.cage_bed_columns[:5])
		cage_df['name'] = 'CAGE'
		chromatics.write_bed(cage_df, cage_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, cage_fn))

	return generators




def generate_features(args, generators, infile, outfile):
	pairs_df = pd.read_csv(infile) # load EPI data

	if args.use_window == 1:
		regions = ["enhancer", "promoter", "window"]
		if "window_name" not in pairs_df.columns:
			pairs_df = insert_window(pairs_df)
	else:
		regions = ["enhancer", "promoter"]

	assert pairs_df.duplicated().sum() == 0
	training_df = chromatics.generate_training(pairs_df, regions, generators, chunk_size = 2**14, n_jobs = 1)

	# save
	training_df.to_csv(outfile, index=False)



# To save memory, split the input dataframe in advance.
def data_split(infile, n_split = 1):
	df = pd.read_csv(infile)
	cnt = math.ceil(len(df) / n_split)
	start = 0
	end = start + cnt
	for i in range(n_split):
		sub_df = df[start:end]
		outfile = infile.replace(".csv", f"_{i}.csv")
		sub_df.to_csv(outfile, index=False)
		start = end
		end = min(start + cnt, len(df))


def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# p.add_argument("--use_config", type=int, default=0)
	p.add_argument("-i", "--infile", help="input file path")	
	p.add_argument("-o", "--outfile", help="output file path")
	p.add_argument("--cell", required=True, help="cell type")
	p.add_argument("--use_window", action="store_true")
	p.add_argument("--data_split", type=int, default=20)

	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()

	print(args)

	os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
	generators = preprocess_features(args)
	data_split(args.infile, args.data_split)
	df = pd.DataFrame()
	for i in range(args.data_split):
		print(f"{i} / {args.data_split}")
		generate_features(args, generators, args.infile.replace(".csv", f"_{i}.csv"), args.outfile.replace(".csv", f"_{i}.csv"))
		sub_df = pd.read_csv(args.outfile.replace(".csv", f"_{i}.csv"))
		df = pd.concat([df, sub_df])
		os.remove(args.infile.replace(".csv", f"_{i}.csv"))
		os.remove(args.outfile.replace(".csv", f"_{i}.csv"))
	df.to_csv(args.outfile, index=False)


		

		

