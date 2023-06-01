import pandas as pd
import argparse
import os
import pulp
import json
import glob



def extract_positive_pairs(args):
	# load input BENGI/TargetFinder
	input_path = os.path.join(os.path.dirname(__file__), "data", args.input, f"{args.cell}.csv")
	df = pd.read_csv(input_path)

	# extract only positive
	positive_df = df[df["label"] == 1]

	# data directory を この~.pyと同じ場所に作成
	outdir = os.path.join(os.path.dirname(__file__), "data", args.input, "positive_only")
	os.makedirs(outdir, exist_ok=True)
	out_path = os.path.join(outdir, f"{args.cell}.csv")

	positive_df.to_csv(out_path, index=False)


def make_bipartiteGraph(args):
	# load positive only
	data_path = os.path.join(os.path.dirname(__file__), "data", args.input, "positive_only", f"{args.cell}.csv")
	df = pd.read_csv(data_path)
	df_by_chrom = df.groupby("enhancer_chrom")

	# cromosome wise
	for chrom, sub_df in df_by_chrom:

		G_from = []
		G_to = []
		G_cap = []

		enhDict_pos = {}
		prmDict_pos = {}


		for _, pair_data in sub_df.iterrows():
			enhName = pair_data["enhancer_name"]
			prmName = pair_data["promoter_name"]

			if enhDict_pos.get(enhName) == None:
				enhDict_pos[enhName] = {}
			if prmDict_pos.get(prmName) == None:
				prmDict_pos[prmName] = {}

			if enhDict_pos[enhName].get(prmName) == None:
				enhDict_pos[enhName][prmName] = 1
			if prmDict_pos[prmName].get(enhName) == None:
				prmDict_pos[prmName][enhName] = 1

		# source => each enhancer
		for enhName in enhDict_pos.keys():
			cap = len(enhDict_pos[enhName])
			G_from.append("source")
			G_to.append(enhName)
			G_cap.append(cap)

		# each promoter => sink
		for prmName in prmDict_pos.keys():
			cap = len(prmDict_pos[prmName])
			G_from.append(prmName)
			G_to.append("sink")
			G_cap.append(cap)

		# each enhancer => each promoter
		enhList = set(sub_df["enhancer_name"].tolist())
		prmList = set(sub_df["promoter_name"].tolist())
		for enhName in enhList:
			assert enhDict_pos.get(enhName) != None
			for prmName in prmList:
				assert prmDict_pos.get(prmName) != None

				enh_start, enh_end = enhName.split("|")[1].split(":")[1].split("-")
				# enh_pos = (int(enh_start) + int(enh_end)) / 2 # TODO how to define enh position
				# prm_pos = prmName.split("|")[0].split(":")[1].split("-")[1] # TODO how to define prm position
				# distance = abs(int(prm_pos) - enh_pos)
				prm_start, prm_end = prmName.split("|")[1].split(":")[1].split("-")
				distance = calc_distance(enh_start, enh_end, prm_start, prm_end)

				# extract pairs in consideration of dmax
				if distance <= args.dmax:
					G_from.append(enhName)
					G_to.append(prmName)
					G_cap.append(1)
		
		bipartiteGraph = pd.DataFrame(
			{
				"from": G_from,
				"to": G_to,
				"cap": G_cap
			},
			index=None
		)

		assert bipartiteGraph.duplicated().sum() == 0

		outdir = os.path.join(os.path.dirname(__file__), "data", args.input, f"dmax_{args.dmax}", "bipartiteGraph", "preprocess")
		os.makedirs(outdir, exist_ok=True)
		out_path = os.path.join(outdir, f"{args.cell}_{chrom}.csv")
		bipartiteGraph.to_csv(out_path, index=False)

def maxflow(args):
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		data_path = os.path.join(os.path.dirname(__file__), "data", args.input, f"dmax_{args.dmax}", "bipartiteGraph", "preprocess", f"{args.cell}_{chrom}.csv")
		if os.path.exists(data_path) == False:
			continue
		df = pd.read_csv(data_path)

		from_list = df["from"].tolist()
		to_list = df["to"].tolist()
		cap_list = df["cap"].tolist()

		# "z" is sum of flow from "source"
		# Maximizing "z" is our goal
		z = pulp.LpVariable("z", lowBound=0)
		problem = pulp.LpProblem("maxflow", pulp.LpMaximize)
		problem += z

		# create variables
		df["Var"] = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=cap_list[i],cat=pulp.LpInteger) for i in df.index]

		# Added constraints on all vertices (flow conservation law)
		for node in set(from_list)|set(to_list):
			if node == "source":
				# sum of flow from "source" == "z"
				fromSource_df = df[df["from"] == node]
				sumFlowFromSource = pulp.lpSum(fromSource_df["Var"])
				problem += sumFlowFromSource == z
			elif node == "sink":
				# sum of flow to "sink" == "z"
				toSink_df = df[df["to"] == node]
				sumFlowToSink = pulp.lpSum(toSink_df["Var"])
				problem += sumFlowToSink == z
			else:
				# sum of flow into a vertex == sum of flow out
				fromNowNode = df[df["from"] == node]
				toNowNode = df[df["to"] == node]
				sumFlowFromNode = pulp.lpSum(fromNowNode["Var"])
				sumFlowToNode = pulp.lpSum(toNowNode["Var"])
				problem += sumFlowFromNode == sumFlowToNode

		# solve
		problem.solve()
		df['Val'] = df.Var.apply(pulp.value)

		assert df.duplicated().sum() == 0

		outdir = os.path.join(os.path.dirname(__file__), "data", args.input, f"dmax_{args.dmax}", "bipartiteGraph", "result")
		os.makedirs(outdir, exist_ok=True)
		out_path = os.path.join(outdir, f"{args.cell}_{chrom}.csv")
		df.to_csv(out_path, index=False)


def get_range_from_name(name):
	start, end = name.split("|")[1].split(":")[1].split("-")
	return int(start), int(end)

def calc_distance(enh_start, enh_end, prm_start, prm_end): # TODO how to define distance???
	enh_pos = (int(enh_start) + int(enh_end)) / 2
	prm_pos = (int(prm_start) + int(prm_end)) / 2
	distance = abs(enh_pos - prm_pos)
	return distance


def concat_CBOEPnegative_and_positive(args):
	# load positive
	positive_path = os.path.join(os.path.dirname(__file__), "data", args.input, "positive_only", f"{args.cell}.csv")
	positive_df = pd.read_csv(positive_path)

	# generate negative from maxflow result
	negative_df = pd.DataFrame(columns=["enhancer_chrom", "promoter_chrom", "from", "to"])
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		maxflow_path = os.path.join(os.path.dirname(__file__), "data", args.input, f"dmax_{args.dmax}", "bipartiteGraph", "result", f"{args.cell}_{chrom}.csv")
		if os.path.exists(maxflow_path) == False:
			continue
		maxflow_df = pd.read_csv(maxflow_path, usecols=["from", "to", "Val"])

		# drop "source" and "sink"
		maxflow_df = maxflow_df[maxflow_df["from"] != "source"]
		maxflow_df = maxflow_df[maxflow_df["to"] != "sink"]
		maxflow_df = maxflow_df[maxflow_df["Val"] == 1]

		maxflow_df.drop("Val", axis=1, inplace=True)
		maxflow_df["enhancer_chrom"] = chrom
		maxflow_df["promoter_chrom"] = chrom

		# concat by chrom
		negative_df = pd.concat([negative_df, maxflow_df], axis=0, ignore_index=True)

	# same format as BENGI/TargetFinder csv
	negative_df.rename(columns={'from': 'enhancer_name', 'to': 'promoter_name'}, inplace=True)
	negative_df["label"] = 0
	negative_df["promoter_chrom"] = negative_df["enhancer_chrom"]
	negative_df["enhancer_start"] = negative_df["enhancer_name"].map(get_range_from_name).map(lambda x: x[0])
	negative_df["enhancer_end"] = negative_df["enhancer_name"].map(get_range_from_name).map(lambda x: x[1])
	if args.input == "BENGI":
		negative_df["promoter_start"] = negative_df["promoter_name"].map(get_range_from_name).map(lambda x: x[0]-1499)
		negative_df["promoter_end"] = negative_df["promoter_name"].map(get_range_from_name).map(lambda x: x[1]+500)
	else:
		negative_df["promoter_start"] = negative_df["promoter_name"].map(get_range_from_name).map(lambda x: x[0])
		negative_df["promoter_end"] = negative_df["promoter_name"].map(get_range_from_name).map(lambda x: x[1])
	negative_df["enhancer_distance_to_promoter"] = abs(((negative_df["enhancer_start"] + negative_df["enhancer_end"]) / 2) - 
	((negative_df["promoter_start"] + negative_df["promoter_end"]) / 2))

	# concat positive and negative
	new_dataset = pd.concat([positive_df, negative_df], axis=0, ignore_index=True)
	assert new_dataset.duplicated().sum() == 0

	# save
	out_path = os.path.join(args.outdir, f"{args.cell}.csv")
	new_dataset.to_csv(out_path, index=False)




def make_new_dataset(args):
	extract_positive_pairs(args)
	make_bipartiteGraph(args)
	maxflow(args)
	concat_CBOEPnegative_and_positive(args)



def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("-input", default="BENGI")
	p.add_argument("--outdir", default="")
	p.add_argument("-dmax", type=int, default=2500000)
	p.add_argument("-cell", type=str, default="GM12878")
	
	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()

	assert args.dmax > 0

	print(f"input {args.input}")
	print(f"max_d {args.dmax}")
	print(f"cell {args.cell}")
	args.outdir = os.path.join(os.path.dirname(__file__), "output", args.input, f"dmax_{args.dmax}")
	os.makedirs(args.outdir, exist_ok=True)
	make_new_dataset(args)

	# for input_file in ["BENGI", "TargetFinder"]:
	# 	for d in [2500000, 5000000, 10000000, 9999999999]:
	# 		for cell in ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]:
	# 			if input_file == "TargetFinder" and cell == "HMEC":
	# 				continue
					
	# 			args.input, args.dmax, args.cell = input_file, d, cell
	# 			print(f"input {args.input}")
	# 			print(f"max_d {args.dmax}")
	# 			print(f"cell {args.cell}")
	# 			args.outdir = os.path.join(os.path.dirname(__file__), "pair_data", args.input, f"CBOEP_{args.dmax}")
	# 			os.makedirs(args.outdir, exist_ok=True)
	# 			make_new_dataset(args)

