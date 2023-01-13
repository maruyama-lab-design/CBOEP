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


from line_notify import line_notify # delete this

INF = 9999999999

# 生成した pair data に features を 追加する
# 完成した training data を用いて XGB で予測 feature importance を得る
# (option) feature importance を用いて 予測実験

def read_bed(x, **kwargs):
	# x = enhancer.bed など
	# python r"" の""内はバックスラッシュが特別な意味を持たない
	return pd.read_csv(x, sep = r'\s+', header = None, index_col = False, **kwargs)

def write_bed(df, fn, **kwargs):
	df = df.copy()
	df.iloc[:, [1, 2]] = df.iloc[:, [1, 2]].astype(int) # ensure coordinates are sorted as integer
	df.sort_values(df.columns.tolist()[:3], inplace = True)
	df.to_csv(fn, sep = '\t', header = False, index = False, **kwargs)

def bedtools(operation, left_input, right_input = None, left_names = None, right_names = None):
	# ___呼び出し部分___
	# bedtools('intersect -wa -wb', chunk_df[region_bed_columns].drop_duplicates(region + '_name'), dataset, right_names = ["chrom", "start", "end", 'dataset', 'signal_value'])
	# ____________________

	# operation = "intersect -wa -wb"
	# left_input = 各行がenhancerの領域情報を表すdataframe など
	# right_input = "peak.bed" など
	# right_names = ["chrom", "start", "end", "dataset", "signal_value"] など

	# if first input is a dataframe, feed via stdin
	if isinstance(left_input, pd.DataFrame):
		# ここを通るはず
		print("left_input OK!!")
		left_input_fn = 'stdin'
	elif isinstance(left_input, str):
		left_input_fn = os.path.abspath(left_input)
	else:
		raise Exception('First input must be DataFrame or filename.')

	# if second input is a dataframe, write to a temp file to be removed later
	right_input_cleanup = False
	if isinstance(right_input, pd.DataFrame):
		right_input_fd, right_input_fn = tempfile.mkstemp()
		right_input_cleanup = True
		write_bed(right_input, right_input_fn)
	elif isinstance(right_input, str):
		# ここを通るはず
		print("right_input OK!!")
		right_input_fn = os.path.abspath(right_input)

	# create command line for one or two argument operations
	if operation.startswith('merge'):
		cmdline = 'bedtools {} "{}"'.format(operation, left_input_fn)
	else:
		# ここを通るはず
		cmdline = 'bedtools {} -a "{}" -b "{}"'.format(operation, left_input_fn, right_input_fn)
		print(cmdline)

	# call bedtools, need shell = True for read permissions
	# bedtools の返り値は、
		# enh(prm/win) chrom,
		# enh(prm/win) start,
		# enh(prm/win) end,
		# enh(prm/win) name,
		# feature chrom,
		# feature start,
		# feature end
		# dataset (feature name)
		# signal_value
	# のカラムになるはずだ...
	p = subprocess.Popen(cmdline, shell = True, stdout = subprocess.PIPE, stdin = subprocess.PIPE, close_fds = True)

	# if needed, write first dataframe in bed format to stdin
	if left_input_fn == 'stdin':
		left_input_buffer = io.StringIO() # ???
		left_input.to_csv(left_input_buffer, sep = '\t', header = False, index = False) # left_input_bifferにleft_inputをcsv保存？？？
		stdout, _ = p.communicate(input = left_input_buffer.getvalue().encode('utf-8'))
	else:
		stdout, _ = p.communicate()
	assert p.returncode == 0

	# if second input was a dataframe written to a temporary file, clean it up
	if right_input_cleanup:
		os.close(right_input_fd)
		os.remove(right_input_fn)
		assert not os.path.exists(right_input_fn)

	# infer column names
	if isinstance(left_input, pd.DataFrame) and left_names is None:
		left_names = left_input.columns.tolist() # chrom, start, end, name
	if isinstance(right_input, pd.DataFrame) and right_names is None:
		right_names = right_input.columns.tolist()

	if operation.startswith('intersect'):
		names = []
		if operation.find('-wa') != -1:
			# 
			names += left_names
		if operation.find('-wb') != -1:
			# 
			names += right_names
		if operation.find('-wo') != -1 or operation.find('-wao') != -1:
			names = left_names + right_names + ['overlap']
		if operation.find('-c') != -1:
			names = left_names + ['count']
		if operation.find('-u') != -1:
			names = left_names
		if operation.find('-loj') != -1:
			names = left_names + right_names
	elif operation.startswith('merge'):
		names = left_names
	elif operation.startswith('closest'):
		names = left_names + right_names
		if operation.find('-d') != -1:
			names.append('distance')
	else:
		names = left_names + right_names

	print(names) # "chrom", "start", "end", "name", "chrom", "start", "end", "dataset", "signal_value"

	# create dataframe from bedtools output stored in stdout
	if len(stdout) == 0:
		return pd.DataFrame(columns = names)
	return read_bed(io.StringIO(stdout.decode('utf-8')), names = names)


def generate_average_signal_features(chunk_df, region, dataset):
	# chunk_df = pair_df の一部
	# region = "enhancer" or "promoter" or "window"
	# dataset = "peak.bed" or "methylation.bed" or "cage.bed"

	assert (chunk_df[region + '_end'] >= chunk_df[region + '_start']).all(), chunk_df[chunk_df[region + '_end'] <= chunk_df[region + '_start']].head()

	# ["enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name"]
	region_bed_columns = ['{}_{}'.format(region, _) for _ in ["chrom", "start", "end", "name"]]

	# bedtools used only here!!!
	# TODO signal_dfがどのような形なのか？
	# > "enh_chrom", "enh_start", "enh_end", "enh_name", "chrom", "start", "end", "dataset", "signal_value"
	signal_df = bedtools('intersect -wa -wb', chunk_df[region_bed_columns].drop_duplicates(region + '_name'), dataset, right_names = ["chrom", "start", "end", 'dataset', 'signal_value'])

	# ["enhancer_name", "enhancer_start", "enhancer_end", "dataset"]
	group_columns = ['{}_{}'.format(region, _) for _ in ['name', 'start', 'end']] + ['dataset']

	# TODO average_signal_dfがどのような形なのか？
	average_signal_df = signal_df.groupby(group_columns, sort = False, as_index = False).aggregate({'signal_value': sum})
	average_signal_df['signal_value'] /= average_signal_df[region + '_end'] - average_signal_df[region + '_start']
	average_signal_df['dataset'] += ' ({})'.format(region)

	# 各行にエンハンサー名、
	# 各カラムにゲノミックフィーチャーの平均値をエンハンサーの長さで割った値
	# が入ったデータフレーム
	return average_signal_df.pivot_table(index = region + '_name', columns = 'dataset', values = 'signal_value')


def generate_chunk_features(pairs_df, regions, generators, chunk_size, chunk_number, max_chunks):
	print(chunk_number, max_chunks - 1)

	chunk_lower_bound = chunk_number * chunk_size
	chunk_upper_bound = chunk_lower_bound + chunk_size
	chunk_df = pairs_df.iloc[chunk_lower_bound:chunk_upper_bound]
	assert 0 < len(chunk_df) <= chunk_size

	# ["enhancer_name", "promoter_name", ("window_name")]
	index_columns = ['{}_name'.format(region) for region in regions]
	features_df = chunk_df[index_columns]
	for region in regions:
		# TODO What is generators??
		# generators = (generate_average_signal_features(関数名), peaks_fn(peak, methylation or cage のパス))
		region_features = [generator(chunk_df, region, dataset) for generator, dataset in generators]
		region_features_df = pd.concat(region_features, axis = 1)
		features_df = pd.merge(features_df, region_features_df, left_on = '{}_name'.format(region), right_index = True, how = 'left')
	return features_df.set_index(index_columns)


def generate_training(pairs_df, regions, generators, chunk_size = 2**16, n_jobs = -1):
	# pair_dfに_chrom, _start, _end, _nameがあるかの確認
	for region in regions:
		region_bed_columns = {'{}_{}'.format(region, _) for _ in ["chrom", "start", "end", "name"]}
		assert region_bed_columns.issubset(pairs_df.columns), pairs_df.columns

	max_chunks = int(np.ceil(len(pairs_df) / chunk_size)) # 何個のチャンクに分かれるか

	results = joblib.Parallel(n_jobs, verbose=1)(
		joblib.delayed(generate_chunk_features)(pairs_df, regions, generators, chunk_size, chunk_number, max_chunks)
		for chunk_number in range(max_chunks)
	)
	# joblib.Parallel()(joblib.delayed(関数名)(関数の引数))
	# 関数の返り値をリストにできるという利点がある．

	features_df = pd.concat(results).fillna(0)
	# print(features_df.head())
	# print(f"column size: {len(features_df.columns)}")
	# print(list(features_df.columns))
	training_df = pd.merge(pairs_df, features_df, left_on = ['{}_name'.format(region) for region in regions], right_index = True)
	# test_train_df = training_df[training_df.index.duplicated(keep=False)]
	# test_train_df.to_csv("./test.csv")

	# check duplicated pair...
	assert len(training_df) == len(training_df[~training_df.index.duplicated()]), print("重複が見つかりました")
	assert training_df.index.is_unique
	assert training_df.columns.is_unique
	return training_df



def add_features_to_trainingData(args):
	# config_fn =
	# cell_line =
	# config =)

	# _fnはfeature bedの書き込み先の相対パス
	peaks_fn = 'features\\peaks.bed.gz'
	methylation_fn = 'features\\methylation.bed.gz'
	cage_fn = 'features\\cage.bed.gz'
	generators = []

	# preprocess peaks
	if os.path.exists(args.peak_root):
		assays = []
		for name, filename, source, accession in pd.read_csv(f'{args.peak_root}/filenames.csv').itertuples(index = False): # 一つ一つのfeatureを読み込み
			columns = ["chrom", "start", "end", "name", "score", "strand", 'signal_value', 'p_value', 'q_value', "peak"]
			columns = columns if filename.endswith('narrowPeak') else columns[:-1] 
			assay_df = read_bed('{}/{}.gz'.format(args.peak_root, filename), names = columns, usecols = ["chrom", "start", "end", "name", "score", "strand", 'signal_value'])
			assay_df['name'] = name # nameはfeatureの名前
			assays.append(assay_df)
		peaks_df = pd.concat(assays, ignore_index = True)
		write_bed(peaks_df, peaks_fn, compression = 'gzip')
		generators.append((generate_average_signal_features, peaks_fn))

	# preprocess methylation
	if os.path.exists(args.methylation_root):
		assays = [read_bed(_, names = ['chrom', 'start', 'end', 'name', "score", "strand", 'thick_start', 'thick_end', 'item_rgb', 'mapped_reads', 'percent_methylated'], usecols = ['chrom', 'start', 'end', 'name', 'mapped_reads', 'percent_methylated']) for _ in glob(f'{args.methylation_root}/*.bed.gz')]
		methylation_df = pd.concat(assays, ignore_index = True).query('mapped_reads >= 10 and percent_methylated > 0')
		methylation_df['name'] = 'Methylation'
		del methylation_df['mapped_reads']
		write_bed(methylation_df, methylation_fn, compression = 'gzip')
		generators.append((generate_average_signal_features, methylation_fn))

	# preprocess cage
	if os.path.exists(args.cage_root):
		cage_df = read_bed(glob(f'{args.cage_root}/*.bed.gz')[0], names = ['chrom', 'start', 'end', 'name', "score", "strand", 'rpkm1', 'rpkm2', 'idr'], usecols = ['chrom', 'start', 'end', 'name', "score"])
		cage_df['name'] = 'CAGE'
		write_bed(cage_df, cage_fn, compression = 'gzip')
		generators.append((generate_average_signal_features, cage_fn))

	# TODO 書き換える
	# feature が そのまま連結されてないか確かめる 上書きならOK
	pairs_df = pd.read_csv(args.in_filename)
	# reshape_for_TargetFinder(pairs_df) # すでにcolumnがある場合は上書き
	assert pairs_df.duplicated().sum() == 0
	if args.use_window == True:
		pairs_df = pairs_df[["enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "label", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name", "window_chrom", "window_start", "window_end", "window_name"]]
		training_df = generate_training(pairs_df, ["enhancer", "promoter", "window"], generators, chunk_size = 2**14, n_jobs = 1)
	else:
		pairs_df = pairs_df[["enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name", "label", "promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]]	
		training_df = generate_training(pairs_df, ["enhancer", "promoter"], generators, chunk_size = 2**14, n_jobs = 1)	
	training_df.to_csv(args.out_filename, index=False)


# ______________________________


def run(args):
	add_features_to_trainingData(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinder実行")
	parser.add_argument("--dataset_name", default="BENGI")
	parser.add_argument("--dataset_type", default="original")
	parser.add_argument("--in_filename", default="")
	parser.add_argument("--out_filename", default="")
	parser.add_argument("--cage_root", help="cage data", default="")
	parser.add_argument("--peak_root", help="peak data", default="")
	parser.add_argument("--methylation_root", help="methylation data", default="")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--use_window", action="store_true")
	args = parser.parse_args()

	args.use_window = True

	for dataset_name in ["TargetFinder"]:
		# for dataset_type in ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", f"maxflow_{INF}"]:
		for dataset_type in ["original"]:
			# for cl in ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]:
			for cl in ["GM12878"]:
				args.dataset_type = dataset_type
				args.dataset_name = dataset_name
				args.cell_line = cl

				if args.use_window == True:
					in_filenames = glob(os.path.join(os.path.dirname(__file__), "epw", "wo_feature", args.dataset_name, args.dataset_type, f"{args.cell_line}*.csv"))
				else:
					in_filenames = glob(os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset_name, args.dataset_type, f"{args.cell_line}*.csv"))

				for in_filename in in_filenames:
					args.in_filename = in_filename
					basename = os.path.splitext(os.path.basename(in_filename))[0] + ".csv"
					if args.use_window == True:
						args.out_filename = os.path.join(os.path.dirname(__file__), "epw", "w_feature", args.dataset_name, args.dataset_type, basename)
					else:
						args.out_filename = os.path.join(os.path.dirname(__file__), "ep", "w_feature", args.dataset_name, args.dataset_type, basename)

					if os.path.exists(args.out_filename) == True:
						continue

					args.cage_root = os.path.join(os.path.dirname(__file__), "features", args.cell_line, "cage")
					args.peak_root = os.path.join(os.path.dirname(__file__), "features", args.cell_line, "peaks")
					args.methylation_root = os.path.join(os.path.dirname(__file__), "features", args.cell_line, "methylation")
					if not os.path.exists(os.path.dirname(args.out_filename)):
						os.makedirs(os.path.dirname(args.out_filename))
					print(f"make {args.out_filename}...")


					# run(args)

					try:
						run(args)
					except Exception as e:
						line_notify(e)
						text = f"{args.dataset_name} {args.dataset_type} {basename} error!!"
						line_notify(text)
					else:
						text = f"{args.dataset_name} {args.dataset_type} {basename} finished!!"
						line_notify(text)

