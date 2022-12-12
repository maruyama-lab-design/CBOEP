from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt

def pr_auc_score(y_true, y_prob):
	precision, recall, _ = mt.precision_recall_curve(y_true, y_prob)
	aupr = mt.auc(recall, precision)
	return aupr


def specifity_score(y_true, y_pred):
	tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

	# NPV
	# return tn / (tn + fn)

	# specificity
	#  TN / TN + FP
	return tn / (tn + fp)


def NPV_score(y_true, y_pred):
	tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

	# NPV
	return tn / (tn + fn)


def result_analysis(dataname_list, datatype_list, region_list, cell_line_list):
	# column_names = ["cell line", "tree", "depth", "learning rate", "AUC", "AUPR", "F", "F\'", "balanced accuracy", "precison", "recall", "specificity", "NPV", "MCC"]
	column_names = ["dataname", "datatype", "region", "cell line", "tree", "depth", "alpha", "balanced acc", "MCC", "TP", "TN", "FP", "FN"]
	data_list = []

	for dataname in dataname_list:
		for datatype in datatype_list:
			for region in region_list:
				for cl in cell_line_list:
					filename_list = glob.glob(os.path.join(os.path.dirname(__file__), "..", region, f"{dataname}_{datatype}", f"{cl}*.csv"))
					for filename in filename_list:
						rn = os.path.basename(filename).removesuffix(".csv")
						_, tree, depth, alpha = rn.split(",")
						df = pd.read_csv(filename)
						true = df["y_test"].to_list()
						prob = df["y_pred"].to_list()

						auc = mt.roc_auc_score(true, prob)
						aupr = pr_auc_score(true, prob)
						pred =  list(map(round, prob))
						true_prime = [int(i == 0) for i in true]
						pred_prime = [int(i == 0) for i in pred]
						f1 = mt.f1_score(true, pred)
						f1_prime = mt.f1_score(true_prime, pred_prime)
						ba = mt.balanced_accuracy_score(true, pred)
						pre = mt.precision_score(true, pred)
						rec = mt.recall_score(true, pred)
						spc = specifity_score(true, pred)
						npv = NPV_score(true, pred)
						mcc = mt.matthews_corrcoef(true, pred)

						tn, fp, fn, tp = mt.confusion_matrix(true, pred).ravel()
						
						data_list.append([dataname, datatype, region, cl, tree, depth, alpha, ba, mcc, tp, tn, fp, fn])


	csv_df = pd.DataFrame(data=data_list, index=None, columns=column_names)
	csv_df.to_csv(f"TargetFinder_result.csv", sep="\t", index=False)
	with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), "result_analysis.xlsx")) as writer:
		csv_df.to_excel(writer, sheet_name=f"TargetFinder tool")

		
		
dataname = ["BENGI", "TargetFinder"]
datatype = ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", "maxflow_9999999999"]
region = ["ep", "epw"]
cl = ["GM12878", "HeLa", "K562", "IMR90", "NHEK"]
result_analysis(dataname, datatype, region, cl)