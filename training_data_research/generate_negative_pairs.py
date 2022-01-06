import pandas as pd
import argparse
import os
import pulp


def download_trainingData(args):
	# data directory を この~.pyと同じ場所に作成
	output_dir = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder")
	os.system(f"mkdir -p {output_dir}")
	# 保存先
	output_path = os.path.join(output_dir, f"{args.cell_line}_train.csv")
	if os.path.exists(output_path):
		return 0

	# training data の url (TargetFinder)からダウンロード
	targetfinder_url = os.path.join(args.targetfinder_root, args.cell_line, "output-ep", "training.csv.gz")
	targetfinder_train_df = pd.read_csv(targetfinder_url,compression='gzip',error_bad_lines=False)

	# 保存
	targetfinder_train_df.to_csv(output_path, index=False)


def extract_positive_pairs(args):
	# data directory を この~.pyと同じ場所に作成
	output_dir = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder", "positive_only")
	os.system(f"mkdir -p {output_dir}")
	# 保存先
	output_path = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder", "positive_only", f"{args.cell_line}_train.csv")
	if os.path.exists(output_path):
		return

	data_path = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder", f"{args.cell_line}_train.csv")
	df = pd.read_csv(data_path)

	# 正例のみを取り出す
	positiveOnly_df = df[df["label"] == 1]
	positiveOnly_df.to_csv(output_path, index=False)


def make_bipartiteGraph(args):
	
	data_path = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder", "positive_only", f"{args.cell_line}_train.csv")
	df = pd.read_csv(data_path, usecols=["enhancer_chrom", "enhancer_name", "promoter_name"])
	df_by_chrom = df.groupby("enhancer_chrom")

	# 染色体ごとに
	for chrom, sub_df in df_by_chrom:

		G_from = []
		G_to = []
		G_cap = []

		enhDict = {}
		prmDict = {}
		# 正例辺を列挙し，dictに保持しておく
		for _, pair_data in sub_df.iterrows():
			enhName = pair_data["enhancer_name"]
			prmName = pair_data["promoter_name"]

			if enhDict.get(enhName) == None:
				enhDict[enhName] = {}
			if prmDict.get(prmName) == None:
				prmDict[prmName] = {}

			if enhDict[enhName].get(prmName) == None:
				enhDict[enhName][prmName] = 1
			if prmDict[prmName].get(enhName) == None:
				prmDict[prmName][enhName] = 1

		# source => 各エンハンサーの容量は，各エンハンサーの正例辺の次数
		for enhName in enhDict.keys():
			cap = len(enhDict[enhName])
			G_from.append("source")
			G_to.append(enhName)
			G_cap.append(cap)
		# 各プロモーター => sinkの容量は，各プロモーターの正例辺の次数
		for prmName in prmDict.keys():
			cap = len(prmDict[prmName])
			G_from.append(prmName)
			G_to.append("sink")
			G_cap.append(cap)

		# 正例辺が貼られていない部分全てに容量1の負例辺を貼る
		enhList = sub_df["enhancer_name"].tolist()
		prmList = sub_df["promoter_name"].tolist()
		for enhName in enhList:
			for prmName in prmList:
				if enhDict[enhName].get(prmName) == None:
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

		output_dir = os.path.join(os.path.dirname(__file__), "bipartiteGraph", args.cell_line, chrom)
		os.system(f"mkdir -p {output_dir}")
		output_path = os.path.join(output_dir, "maxFlow_preprocess.csv")
		bipartiteGraph.to_csv(output_path, index=False)

def maximumFlow(args):
	# 線計画法にて最大流問題を解く
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		data_path = os.path.join(os.path.dirname(__file__), "bipartiteGraph", args.cell_line, chrom, "maxFlow_preprocess.csv")
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

		# 解を求める
		result = problem.solve()
		# LpStatusが`optimal`なら最適解が得られた事になる
		print(pulp.LpStatus[result])
		# 目的関数の値
		print(pulp.value(problem.objective))
		# 'Var'変数の結果の値をまとめて'Val'列にコピーしている
		df['Val'] = df.Var.apply(pulp.value)

		output_path = os.path.join(os.path.dirname(__file__), "bipartiteGraph", args.cell_line, chrom, "maxFlow_result.csv")
		df.to_csv(output_path, index=False)


def assemble_new_trainingData(args):
	# TargetFinderのpositive_only学習データと，maxFlowで作った染色体毎のnegative学習データを結合する

	# positive_onlyの学習データをダウンロード
	positive_data_path = os.path.join(os.path.dirname(__file__), "training_data", "TargetFinder", "positive_only", f"{args.cell_line}_train.csv")
	positive_only_df = pd.read_csv(positive_data_path, usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"])

	# negativeの学習データを作成
	maximumFlow_result_df = pd.DataFrame(columns=["enhancer_chrom", "from", "to"])
	chromList = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
	for chrom in chromList:
		maximumFlow_result_path = os.path.join(os.path.dirname(__file__), "bipartiteGraph", args.cell_line, chrom, "maxFlow_result.csv")
		maximumFlow_result_subdf = pd.read_csv(maximumFlow_result_path, usecols=["from", "to", "Val"])

		# いらない行の削除
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["from"] != "source"]
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["to"] != "sink"]
		maximumFlow_result_subdf = maximumFlow_result_subdf[maximumFlow_result_subdf["Val"] == 1]

		maximumFlow_result_subdf.drop("Val", axis=1, inplace=True)
		maximumFlow_result_subdf["enhancer_chrom"] = chrom

		# 結合していく
		maximumFlow_result_df = maximumFlow_result_df.append(maximumFlow_result_subdf, ignore_index=True)

	# TargetFinderの学習データセットと同じ形式にする
	maximumFlow_result_df.rename(columns={'from': 'enhancer_name', 'to': 'promoter_name'}, inplace=True)
	maximumFlow_result_df["label"] = 0

	# 正例と負例をconcat
	new_trainingData_df = pd.concat([positive_only_df, maximumFlow_result_df], axis=0, ignore_index=True)

	# 保存先のディレクトリを作成し，保存
	output_dir = os.path.join(os.path.dirname(__file__), "training_data", "new")
	os.system(f"mkdir -p {output_dir}")
	output_path = os.path.join(output_dir, f"{args.cell_line}_train.csv")
	new_trainingData_df.to_csv(output_path, index=False)


def make_new_trainingData(args):
	download_trainingData(args)
	extract_positive_pairs(args)
	make_bipartiteGraph(args)
	maximumFlow(args)
	assemble_new_trainingData(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--targetfinder_root", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	args = parser.parse_args()

	cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
	for cell_line in cell_line_list:
		args.cell_line = cell_line
		make_new_trainingData(args)


