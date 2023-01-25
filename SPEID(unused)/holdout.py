import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import build_model as bm
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import argparse
import glob
import pickle


def hold_out(args):
	chr2dataset = None
	with open(os.path.join(os.path.dirname(__file__), "chrom2dataset", args.dataname, args.datatype, f"{args.cell_line}_chrom2dataset.pkl"), "rb") as rf:
		chr2dataset = pickle.load(rf)

	all_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
	train_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15"]
	valid_chroms = ["chr16", "chr17", "chr18"]
	test_chroms =  ["chr19", "chr20", "chr21", "chr22", "chrX"]

	output_path = os.path.join(args.output_dir, f"{args.cell_line}_foldout.csv")

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


	model = bm.build_model()
	model.compile(
		loss="binary_crossentropy",
		optimizer=Adam(learning_rate=1e-5),
	)
	# print(model.summary())

	checkpoint_path = os.path.join(os.path.dirname(__file__), "model", args.dataname, args.datatype, f"{args.cell_line}_model.h5")
	os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
	csv_logger = CSVLogger(os.path.join(args.output_dir, f"{args.cell_line}_history.csv"))
	modelCheckpoint = ModelCheckpoint(		
		filepath = checkpoint_path,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=False,
		mode='min',
		period=1
	)
	earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
	print("training...")
	model.fit(
		x=[X_train_enh, X_train_prm], 
		y=[Y_train],
		epochs=args.epochs,
		batch_size=args.batch_size,
		# class_weight={1: 20, 0:1},
		callbacks=[modelCheckpoint, earlystopper, csv_logger],
		validation_data=([X_valid_enh, X_valid_prm], [Y_valid])
	)
	
	print("model loading...")
	model.load_weights(checkpoint_path)

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
	parser.add_argument("--dataname", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--dataytpe", help="どのデータセットを使うか", default="original")
	parser.add_argument("--cell_line", help="細胞株", default="GM12878")
	parser.add_argument("--epochs", type=int, default=32)
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--output_dir", default="")
	args = parser.parse_args()

	# for cell_line in ["GM12878", "K562", "HeLa", "HUVEC", "IMR90", "NHEK"]:
	for cell_line in ["GM12878"]:
		for dataset in ["TargetFinder", "BENGI"]:
			for datatype in ["original", "maxflow"]:
				for epochs in [32]:
					for batch_size in [32]:

						args.cell_line = cell_line
						args.dataname = dataset
						args.datatype = datatype
						args.epochs = epochs
						args.batch_size = batch_size
						args.output_dir = os.path.join(os.path.dirname(__file__), "result", args.dataname, args.datatype)


						os.makedirs(args.output_dir, exist_ok=True)

						hold_out(args)
