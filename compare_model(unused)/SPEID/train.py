import build_model as bm
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import argparse
import os


def seq2vector(seq, length):
	dic = {
		"a":np.array([1, 0, 0, 0]),
		"c":np.array([0, 1, 0, 0]),
		"g":np.array([0, 0, 1, 0]),
		"t":np.array([0, 0, 0, 1]),
		"n":np.array([0, 0, 0, 0])
	}

	vector = np.zeros((length, 4))
	for i in range(length):
		acgt = ""
		if i < len(seq):
			acgt = seq[i]
		else:
			acgt = "n"
		vector[i] = dic[acgt]

	return vector


def get_name2seq(args):
	# return dict
	# which return sequence(acgt) from sequence_name
	name2seq_enh, name2seq_prm = {}, {}
	key_cash = ""
	with open(f"/home/koga/Koga_code-1/compare_model/SPEID/sequence/{args.cell_line}/enhancers.fa") as f_enh:
		for seq in f_enh.readlines():
			if seq[0] == ">":
				key_cash = seq[1:].strip()
			else:
				# key_cashは初めて出るはずである
				assert key_cash not in name2seq_enh
				name2seq_enh[key_cash] = seq.lower().strip()
	
	with open(f"/home/koga/Koga_code-1/compare_model/SPEID/sequence/{args.cell_line}/promoters.fa") as f_prm:
		for seq in f_prm.readlines():
			if seq[0] == ">":
				key_cash = seq[1:].strip()
			else:
				# key_cashは初めて出るはずである
				assert key_cash not in name2seq_prm
				name2seq_prm[key_cash] = seq.lower().strip()

	# print(X_enhancers_seq)
	# print(X_promoters_seq)

	return name2seq_enh, name2seq_prm


def get_name2vec(args):
	# return dict
	# which return vector from sequence_name

	name2vec = {}
	name2seq_enh, name2seq_prm = get_name2seq(args)
	for name, seq in name2seq_enh.items():
		name2vec[name] = seq2vector(seq, 3000)
	for name, seq in name2seq_prm.items():
		name2vec[name] = seq2vector(seq, 2000)
	return name2vec


def get_chr2dataset(args):
	# return dict
	# which return dataset from chromosome
	# dataset has keys such as "X_enhancers", "X_promoters", and "label"

	chr2dataset = {
		"chr1": {
			"X_enhancers":[],
			"X_promoters":[],
			"labels":[]
		},
		"chr2": {
			"X_enhancers":[],
			"X_promoters":[],
			"labels":[]
		},
		# 続く．．．
	}

	name2vec = get_name2vec(args)

	df = pd.read_csv(f"/home/koga/Koga_code-1/compare_model/SPEID/../training_data/{args.dataset}/{args.cell_line}_train.csv")
	for _, data in df.iterrows():
		enh_name = data["enhancer_name"]
		prm_name = data["promoter_name"]

		enh_vector = name2vec[enh_name]
		prm_vector = name2vec[prm_name]
		label = data["label"]
		chromosome = data["enhancer_chrom"]

		if chromosome not in chr2dataset:
			chr2dataset[chromosome] = {
				"X_enhancers":[],
				"X_promoters":[],
				"labels":[]
			}

		chr2dataset[chromosome]["X_enhancers"].append(enh_vector)
		chr2dataset[chromosome]["X_promoters"].append(prm_vector)
		chr2dataset[chromosome]["labels"].append(label)

	return chr2dataset


def chromosomal_cv(args):
	chr2dataset = get_chr2dataset(args)

	all_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
	test_chroms = [["chr1", "chr2"], ["chr3", "chr4"], ["chr5", "chr6"], ["chr7", "chr8"], ["chr9", "chr10"], ["chr11", "chr12"], ["chr13", "chr14"], ["chr15", "chr16"], ["chr17", "chr18"], ["chr19", "chr20"], ["chr21", "chr22"]]

	for fold, test_chrom in enumerate(test_chroms):
		output_path = os.path.join(args.output_dir, f"fold_{fold+1}.csv")
		print(f"fold {fold+1}...")
		print(f"test chromosome : {test_chrom}")
		test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]

		X_train_enh, X_test_enh = [], []
		X_train_prm, X_test_prm = [], []
		Y_train, Y_test = [], []


		for chromosome in all_chroms:
			if chromosome == test_chrom1 or chromosome == test_chrom2:
				X_test_enh += chr2dataset[chromosome]["X_enhancers"]
				X_test_prm += chr2dataset[chromosome]["X_promoters"]
				Y_test += chr2dataset[chromosome]["labels"]
			else:
				X_train_enh += chr2dataset[chromosome]["X_enhancers"]
				X_train_prm += chr2dataset[chromosome]["X_promoters"]
				Y_train += chr2dataset[chromosome]["labels"]

		
		X_train_enh = np.array(X_train_enh)
		X_train_prm = np.array(X_train_prm)
		X_test_enh = np.array(X_test_enh)
		X_test_prm = np.array(X_test_prm)
		Y_train = np.array(Y_train)
		Y_test = np.array(Y_test)

		assert len(X_train_enh) == len(X_train_prm)
		assert len(X_train_enh) == len(Y_train)

		assert len(X_test_enh) == len(X_test_prm)
		assert len(X_test_enh) == len(Y_test)

		print(f"train enhancers sample size: {X_train_enh.shape}")
		print(f"train promoters sample size: {X_train_prm.shape}")

		# type check
		print(type(X_train_enh))
		print(X_train_enh.shape)
		print(type(X_train_enh[0]))
		print(X_train_enh[0].shape)

		model = bm.build_model()
		model.compile(
			loss="binary_crossentropy",
			optimizer=Adam(learning_rate=1e-5),
		)
		# print(model.summary())

		csv_logger = CSVLogger(os.path.join(args.output_dir, f"fold{fold}_log"))
		modelCheckpoint = ModelCheckpoint(
			filepath = f'/home/koga/Koga_code-1/compare_model/SPEID/model/{args.cell_line}_{fold+1}.h5',
			monitor='loss',
			verbose=1,
			save_best_only=True,
			save_weights_only=False,
			mode='min',
			save_freq=1
		)
		earlystopper = EarlyStopping(monitor='loss', patience=50, verbose=1)
		print("training...")
		model.fit(
			[X_train_enh, X_train_prm], 
			[Y_train],
			epochs=args.epochs,
			batch_size=args.batch_size,
			# class_weight={1: 20, 0:1},
			callbacks=[modelCheckpoint, earlystopper, csv_logger]
		)

		model.load_weights(f'/home/koga/Koga_code-1/compare_model/SPEID/model/{args.cell_line}_{fold+1}.h5')
		print("testing...")
		y_pred = model.predict(
			[X_test_enh, X_test_prm],
			batch_size=args.batch_size,
			verbose=1
		)

		result_df = pd.DataFrame(
			{
				"y_test": Y_test,
				"y_pred": np.reshape(y_pred, [-1])
			},
			index=None
		)
		result_df.to_csv(output_path)


def hold_out(args):
	chr2dataset = get_chr2dataset(args)

	all_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
	train_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16"]
	valid_chroms = ["chr17", "chr18"]
	test_chroms =  ["chr19", "chr20", "chr21", "chr22", "chrX"]

	output_path = os.path.join(args.output_dir, f"foldout.csv")
	test_chrom1, test_chrom2 = test_chrom[0], test_chrom[1]

	X_train_enh, X_valid_enh, X_test_enh = [], [], []
	X_train_prm, X_valid_prm, X_test_prm = [], [], []
	Y_train, Y_valid, Y_test = [], [], []


	for chromosome in all_chroms:
		if chromosome in test_chroms:
			X_test_enh += chr2dataset[chromosome]["X_enhancers"]
			X_test_prm += chr2dataset[chromosome]["X_promoters"]
			Y_test += chr2dataset[chromosome]["labels"]
		elif chromosome in train_chroms:
			X_train_enh += chr2dataset[chromosome]["X_enhancers"]
			X_train_prm += chr2dataset[chromosome]["X_promoters"]
			Y_train += chr2dataset[chromosome]["labels"]
		elif chromosome in valid_chroms:
			X_valid_enh += chr2dataset[chromosome]["X_enhancers"]
			X_valid_prm += chr2dataset[chromosome]["X_promoters"]
			Y_valid += chr2dataset[chromosome]["labels"]

	
	X_train_enh = np.array(X_train_enh)
	X_train_prm = np.array(X_train_prm)
	Y_train = np.array(Y_train)

	X_valid_enh = np.array(X_valid_enh)
	X_valid_prm = np.array(X_valid_prm)
	Y_valid = np.array(Y_valid)

	X_test_enh = np.array(X_test_enh)
	X_test_prm = np.array(X_test_prm)
	Y_test = np.array(Y_test)

	assert len(X_train_enh) == len(X_train_prm)
	assert len(X_train_enh) == len(Y_train)

	assert len(X_valid_enh) == len(X_valid_prm)
	assert len(X_valid_enh) == len(Y_valid)

	assert len(X_test_enh) == len(X_test_prm)
	assert len(X_test_enh) == len(Y_test)

	print(f"train enhancers sample size: {X_train_enh.shape}")
	print(f"train promoters sample size: {X_train_prm.shape}")

	print(f"valid enhancers sample size: {X_valid_enh.shape}")
	print(f"valid promoters sample size: {X_valid_prm.shape}")

	# type check
	# print(type(X_train_enh))
	# print(X_train_enh.shape)
	# print(type(X_train_enh[0]))
	# print(X_train_enh[0].shape)

	model = bm.build_model()
	model.compile(
		loss="binary_crossentropy",
		optimizer=Adam(learning_rate=1e-5),
	)
	# print(model.summary())

	csv_logger = CSVLogger(os.path.join(args.output_dir, f"holdout_log"))
	modelCheckpoint = ModelCheckpoint(		
		filepath = os.path.join(os.path.dirname(__file__), "model", "model-{epoch:02d}-{val_loss:.2f}.h5"),
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=False,
		mode='min',
		save_freq=1
	)
	earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
	print("training...")
	history = model.fit(
		x=[X_train_enh, X_train_prm], 
		y=[Y_train],
		epochs=args.epochs,
		batch_size=args.batch_size,
		# class_weight={1: 20, 0:1},
		callbacks=[modelCheckpoint, earlystopper, csv_logger],
		validation_data=([X_valid_enh, X_valid_prm], [Y_valid])
	)

	model.load_weights(os.path.join(os.path.dirname(__file__), "model", "model-{epoch:02d}-{val_loss:.2f}.h5"))
	print("testing...")
	y_pred = model.predict(
		[X_test_enh, X_test_prm],
		batch_size=args.batch_size,
		verbose=1
	)

	result_df = pd.DataFrame(
		{
			"y_test": Y_test,
			"y_pred": np.reshape(y_pred, [-1])
		},
		index=None
	)
	result_df.to_csv(output_path)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="EP2vec")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--way_of_cv", help="染色体毎かランダムか", choices=["chromosomal", "random"], default="chromosomal")
	parser.add_argument("--epochs", type=int, default=32)
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--output_dir", default="")
	args = parser.parse_args()

	for cell_line in ["GM12878", "K562", "HeLa-S3", "HUVEC", "IMR90", "NHEK"]:
		for dataset in ["new", "EP2vec"]:
			for epochs in [512]:
				for batch_size in [100]:

					args.dataset = dataset
					args.cell_line = cell_line
					args.epochs = epochs
					args.batch_size = batch_size

					args.output_dir = os.path.join(os.path.dirname(__file__), "result", args.cell_line, args.dataset, f"epochs_{args.epochs}", f"batch_size_{args.batch_size}")
					print(args.output_dir)
					# if os.path.exists(args.output_dir):
					# 	print("continue...")
					# 	continue
					os.system(f"mkdir -p {args.output_dir}")

					# chromosomal_cv(args)
					hold_out(args)

# test_train()
# seq2vector("aaaaaaaa")