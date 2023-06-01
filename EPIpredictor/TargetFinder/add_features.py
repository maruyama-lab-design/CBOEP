# Here, features are added to the pair data.
# The addition of the window regions is also executed here.

#!/usr/bin/env python

import chromatics
import os
import pandas as pd
import sys

from glob import glob
import math
import argparse

import json


def preprocess_features(args):

	peaks_fn = 'peaks.bed.gz'
	methylation_fn = 'methylation.bed.gz'
	cage_fn = 'cage.bed.gz'
	generators = []

	# preprocess peaks
	peaks_dir = os.path.join(os.path.dirname(__file__), "features", args.cell, "peaks")
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
	methylation_dir = os.path.join(os.path.dirname(__file__), "features", args.cell, "methylation")
	if os.path.exists(methylation_dir):
		print(f"preprocess methylation...")
		assays = [chromatics.read_bed(_, names = chromatics.methylation_bed_columns, usecols = chromatics.generic_bed_columns + ['mapped_reads', 'percent_methylated']) for _ in glob(os.path.join(methylation_dir, f"*.bed.gz"))]
		methylation_df = pd.concat(assays, ignore_index = True).query('mapped_reads >= 10 and percent_methylated > 0')
		methylation_df['name'] = 'Methylation'
		del methylation_df['mapped_reads']
		chromatics.write_bed(methylation_df, methylation_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, methylation_fn))

	# preprocess cage
	cage_dir = os.path.join(os.path.dirname(__file__), "features", args.cell, "cage")
	if os.path.exists(cage_dir):
		print(f"preprocess cage...")
		cage_df = chromatics.read_bed(glob(os.path.join(cage_dir, f"*.bed.gz"))[0], names = chromatics.cage_bed_columns, usecols = chromatics.cage_bed_columns[:5])
		cage_df['name'] = 'CAGE'
		chromatics.write_bed(cage_df, cage_fn, compression = 'gzip')
		generators.append((chromatics.generate_average_signal_features, cage_fn))

	return generators




def generate_features(args, generators, infile, outfile):



	if args.use_window:
		regions = ["enhancer", "promoter", "window"]
	else:
		regions = ["enhancer", "promoter"]

	# generate features 
	pairs_df = pd.read_csv(infile) # load pair
	assert pairs_df.duplicated().sum() == 0
	training_df = chromatics.generate_training(pairs_df, regions, generators, chunk_size = 2**14, n_jobs = 1)

	# save
	# training_df.to_hdf('training.h5', 'training', mode = 'w', complevel = 1, complib = 'zlib')

	# Koga
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



## こっから...

# if cell_line == "HeLa-S3":
#     cell_line = "HeLa"


# infile = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/epw/wo_feature/{dataname}/{datatype}/{cell_line}.csv"
# n_split = 10

# data_split(infile, n_split=n_split)

# for i in range(n_split):
#     print(f"{i} 番目...")
#     infile = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/epw/wo_feature/{dataname}/{datatype}/{cell_line}_{i}.csv"
#     outfile = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/epw/w_feature/{dataname}/{datatype}/{cell_line}_{i}.csv"
#     os.makedirs(os.path.dirname(outfile), exist_ok=True)
#     generate_features(infile, outfile)

# infile = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/ep/wo_feature/{dataname}/{datatype}/{cell_line}.csv"
# outfile = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/ep/w_feature/{dataname}/{datatype}/{cell_line}.csv"
# os.makedirs(os.path.dirname(outfile), exist_ok=True)
# generate_features(infile, outfile)


def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--cell", default="GM12878")
	p.add_argument("--data", default="BENGI")
	p.add_argument("--NIMF_max_d", type=int, default=2500000)
	p.add_argument("--use_window", type=bool, default=True)
	p.add_argument("--data_split", type=int, default=1)

	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()

	config = json.load(open(os.path.join(os.path.dirname(__file__), "add_features_opt.json")))
	args.cell = config["cell"]
	args.data = config["data"]
	args.NIMF_max_d = config["NIMF_max_d"]
	args.use_window = config["use_window"]
	args.data_split = config["data_split"]

	print(args)

	if args.NIMF_max_d == -1:
		infile = os.path.join(os.path.dirname(__file__), "..", "pair_data", args.data, "original", f"{args.cell}.csv")
		if args.use_window:
			outfile = os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, "original", "epw", f"{args.cell}.csv")
		else:
			outfile = os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, "original", "ep", f"{args.cell}.csv")
	else:
		infile = os.path.join(os.path.dirname(__file__), "..", "pair_data", args.data, f"NIMF_{args.NIMF_max_d}", f"{args.cell}.csv")
		if args.use_window:
			outfile = os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "epw", f"{args.cell}.csv")
		else:
			outfile = os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "ep", f"{args.cell}.csv")

	
	os.makedirs(os.path.dirname(outfile), exist_ok=True)
	generators = preprocess_features(args)
	data_split(infile, args.data_split)
	df = pd.DataFrame()
	for i in range(args.data_split):
		print(f"{i} / {args.data_split}")
		generate_features(args, generators, infile.replace(".csv", f"_{i}.csv"), outfile.replace(".csv", f"_{i}.csv"))
		sub_df = pd.read_csv(outfile.replace(".csv", f"_{i}.csv"))
		df = pd.concat([df, sub_df])
		os.remove(infile.replace(".csv", f"_{i}.csv"))
		os.remove(outfile.replace(".csv", f"_{i}.csv"))
	df.to_csv(outfile, index=False)


		

		

