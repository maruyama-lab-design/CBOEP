from numpy import NaN
import pandas as pd
import argparse
import os
import pulp

import glob


INF = 9999999999


def extract_positive_pairs(args):
	# data directory を この~.pyと同じ場所に作成
	output_dir = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, "original", "positive_only")
	os.makedirs(output_dir, exist_ok=True)
	# 保存先
	output_path = os.path.join(output_dir, args.filename)


	data_path = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, "original", args.filename)
	df = pd.read_csv(data_path)

	# 正例のみを取り出す
	positiveOnly_df = df[df["label"] == 1]
	positiveOnly_df.to_csv(output_path, index=False)


def make_bipartiteGraph(args):
	
	# data_path = os.path.join(os.path.dirname(__file__), "original", "positive_only", args.filename)
	data_path = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, "original", args.filename)
	df = pd.read_csv(data_path)
	df_by_chrom = df.groupby("enhancer_chrom")

	# 染色体ごとに
	for chrom, sub_df in df_by_chrom:

		G_from = []
		G_to = []
		G_cap = []

		enhancerDict_pos = {}
		promoterDict_pos = {}

		for _, pair_data in sub_df.iterrows():
			enhancerName = pair_data["enhancer_name"]
			promoterName = pair_data["promoter_name"]


			if pair_data["label"] == 1:
				if enhancerDict_pos.get(enhancerName) == None:
					enhancerDict_pos[enhancerName] = {}
				if promoterDict_pos.get(promoterName) == None:
					promoterDict_pos[promoterName] = {}

				if enhancerDict_pos[enhancerName].get(promoterName) == None:
					enhancerDict_pos[enhancerName][promoterName] = 1
				if promoterDict_pos[promoterName].get(enhancerName) == None:
					promoterDict_pos[promoterName][enhancerName] = 1

		# source => 各エンハンサーの容量は，各エンハンサーの正例辺の次数
		for enhancerName in enhancerDict_pos.keys():
			cap = len(enhancerDict_pos[enhancerName])
			G_from.append("source")
			G_to.append(enhancerName)
			G_cap.append(cap)

		# 各プロモーター => sinkの容量は，各プロモーターの正例辺の次数
		for promoterName in promoterDict_pos.keys():
			cap = len(promoterDict_pos[promoterName])
			G_from.append(promoterName)
			G_to.append("sink")
			G_cap.append(cap)


		# 正例辺が貼られていない部分全てに容量1の負例辺を貼る
		enhancerList = set(sub_df["enhancer_name"].tolist())
		promoterList = set(sub_df["promoter_name"].tolist())
		for enhancerName in enhancerList:
			for promoterName in promoterList:
				if enhancerDict_pos.get(enhancerName) != None and enhancerDict_pos[enhancerName].get(promoterName) == None:


					enh_start, enh_end = map(int, enhancerName.split("|")[1].split(":")[1].split("-"))
					assert enh_end > enh_start, enhancerName
					prm_pos = promoterName.split("|")[1].split(":")[1].split("-")
					prm_start = int(prm_pos[0])
					prm_end = int(prm_pos[1])
					if args.dataset == "BENGI":
						prm_start -= 1499
						prm_end += 500
					assert prm_end > prm_start, promoterName
					distance = min(abs(enh_start - prm_end), abs(prm_start - enh_end))

					if distance <= args.max_distance:
						G_from.append(enhancerName)
						G_to.append(promoterName)
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

		output_dir = os.path.join(os.path.dirname(__file__), args.dataset, "ep", "wo_feature",  f"maxflow_{args.max_distance}", "bipartiteGraph", "preprocess", chrom)
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, args.filename)
		bipartiteGraph.to_csv(output_path, index=False)

def maximumFlow(args):
	# 線計画法にて最大流問題を解く
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		data_path = os.path.join(os.path.dirname(__file__), args.dataset, "ep", "wo_feature",  f"maxflow_{args.max_distance}", "bipartiteGraph", "preprocess", chrom, args.filename)
		if os.path.exists(data_path) == False:
			continue
		df = pd.read_csv(data_path)

		from_list = df["from"].tolist()
		to_list = df["to"].tolist()
		cap_list = df["cap"].tolist()

		z = pulp.LpVariable("z", lowBound=0) # このzはsourceから流出するflow量であり，最大化する目的関数でもある.
		problem = pulp.LpProblem("最大流問題", pulp.LpMaximize)
		problem += z # 目的関数を設定
		# 大量の変数を作る
		df["Var"] = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=cap_list[i],cat=pulp.LpInteger) for i in df.index]

		# 全頂点に関する制約条件を追加していく（保存則）
		for node in set(from_list)|set(to_list):
			if node == "source":
				# sourceからのflowの和は変数zに等しい
				fromSource_df = df[df["from"] == node]
				sumFlowFromSource = pulp.lpSum(fromSource_df["Var"])
				problem += sumFlowFromSource == z
			elif node == "sink":
				# sinkのflowの和は変数zに等しい
				toSink_df = df[df["to"] == node]
				sumFlowToSink = pulp.lpSum(toSink_df["Var"])
				problem += sumFlowToSink == z
			else:
				# ある頂点に流入するflowと流出するflowは等しい
				fromNowNode = df[df["from"] == node]
				toNowNode = df[df["to"] == node]
				sumFlowFromNode = pulp.lpSum(fromNowNode["Var"])
				sumFlowToNode = pulp.lpSum(toNowNode["Var"])
				problem += sumFlowFromNode == sumFlowToNode


		# solver_list = pulp.listSolvers(onlyAvailable=True)
		# print(solver_list)
		# 解を求める
		result = problem.solve()
		# LpStatusが`optimal`なら最適解が得られた事になる
		# print(pulp.LpStatus[result])
		# 目的関数の値
		# print(pulp.value(problem.objective))
		# 'Var'変数の結果の値をまとめて'Val'列にコピーしている
		df['Val'] = df.Var.apply(pulp.value)

		assert df.duplicated().sum() == 0

		output_dir = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset,  f"maxflow_{args.max_distance}", "bipartiteGraph", "result", chrom)
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, args.filename)
		df.to_csv(output_path, index=False)


def get_range_from_name(name):
	# name = GM12878|chr1:9685722-9686400
	start, end = name.split("|")[-1].split(":")[-1].split("-")
	return int(start), int(end)


def assemble_new_trainingData(args):
	#positive_only学習データと，maxFlowで作った染色体毎のnegative学習データを結合する

	# positive_onlyの学習データをダウンロード
	positive_data_path = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, "original", "positive_only", args.filename)
	positive_only_df = pd.read_csv(positive_data_path, usecols=["label","enhancer_distance_to_promoter","enhancer_chrom","enhancer_start","enhancer_end","enhancer_name","promoter_chrom","promoter_start","promoter_end","promoter_name"])

	# negativeの学習データを作成
	maximumFlow_result_df = pd.DataFrame(columns=["enhancer_chrom", "promoter_chrom", "from", "to"])
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		maximumFlow_result_path = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset,  f"maxflow_{args.max_distance}", "bipartiteGraph", "result", chrom, args.filename)
		if os.path.exists(maximumFlow_result_path) == False:
			continue
		maximumFlow_result_subdf = pd.read_csv(maximumFlow_result_path, usecols=["from", "to", "Val"])

		# いらない行の削除
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["from"] != "source"]
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["to"] != "sink"]
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["Val"] == 1]

		maximumFlow_result_subdf.drop("Val", axis=1, inplace=True)
		maximumFlow_result_subdf["enhancer_chrom"] = chrom

		# 結合していく
		maximumFlow_result_df = pd.concat([maximumFlow_result_df, maximumFlow_result_subdf], ignore_index=True)
		# maximumFlow_result_df = maximumFlow_result_df.concat(maximumFlow_result_subdf)

	# 元々の学習データセットと同じ形式にする
	maximumFlow_result_df.rename(columns={'from': 'enhancer_name', 'to': 'promoter_name'}, inplace=True)
	maximumFlow_result_df["label"] = 0
	maximumFlow_result_df["promoter_chrom"] = maximumFlow_result_df["enhancer_chrom"]
	# maximumFlow_result_df["end_start"], maximumFlow_result_df["enhancer_end"] = maximumFlow_result_df["enhancer_name"].map(get_range_from_name)
	maximumFlow_result_df["enhancer_start"] = maximumFlow_result_df["enhancer_name"].map(get_range_from_name).map(lambda x: x[0])
	maximumFlow_result_df["enhancer_end"] = maximumFlow_result_df["enhancer_name"].map(get_range_from_name).map(lambda x: x[1])


	if args.dataset == "BENGI":
		maximumFlow_result_df["promoter_start"] = maximumFlow_result_df["promoter_name"].map(get_range_from_name).map(lambda x: x[0]-1499)
		maximumFlow_result_df["promoter_end"] = maximumFlow_result_df["promoter_name"].map(get_range_from_name).map(lambda x: x[1]+500)
	else:
		maximumFlow_result_df["promoter_start"] = maximumFlow_result_df["promoter_name"].map(get_range_from_name).map(lambda x: x[0])
		maximumFlow_result_df["promoter_end"] = maximumFlow_result_df["promoter_name"].map(get_range_from_name).map(lambda x: x[1])

	# maximumFlow_result_df["enhancer_distance_to_promoter"] = max(abs(maximumFlow_result_df["promoter_start"] - maximumFlow_result_df["enhancer_end"]), abs(maximumFlow_result_df["enhancer_start"] - maximumFlow_result_df["promoter_end"]))
	maximumFlow_result_df["enhancer_distance_to_promoter"] = "NaN"
	

	# 正例と負例をconcat
	new_trainingData_df = pd.concat([positive_only_df, maximumFlow_result_df], axis=0, ignore_index=True)

	assert new_trainingData_df.duplicated().sum() == 0, new_trainingData_df[new_trainingData_df.duplicated()]

	# 保存先のディレクトリを作成し，保存
	output_dir = os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, f"maxflow_{args.max_distance}")
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, args.filename)
	new_trainingData_df.to_csv(output_path, index=False, sep=",")

	print(f"pos : neg = {len(positive_only_df)};{len(maximumFlow_result_df)}")


def make_new_trainingData(args):
	extract_positive_pairs(args)
	make_bipartiteGraph(args)
	maximumFlow(args)
	assemble_new_trainingData(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--filename", default="")
	parser.add_argument("--dataset", default="")
	parser.add_argument("--max_distance", type=int, default=INF)
	args = parser.parse_args()

	for dataset in ["BENGI", "TargetFinder"]:
		for cl in ["GM12878", "HeLa", "K562", "IMR90", "NHEK"]:
			for distance in [2500000, 5000000, 10000000, INF]:
				args.dataset = dataset
				args.max_distance = distance
				files = glob.glob(os.path.join(os.path.dirname(__file__), "ep", "wo_feature", args.dataset, "original", f"{cl}*.csv"))
				for file in files:
					args.filename = os.path.basename(file)
					print(f"make maxflow based dataset from {args.filename}...")
					make_new_trainingData(args)


