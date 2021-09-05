import pandas as pd
from tqdm import tqdm

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
# まだ未完成
#--------------


def make_region_label_table(cl, bin_length):
	#-----説明-----
	# enhancer, promoter それぞれが、どの neighbor 領域に含まれているかを表すテーブルデータを作成する.
	# 2make_region_table で作られるテーブルデータが必要

		# テーブルデータの例
			#	neighbor_id		chr   	enhancer				promoter
			#	:
			#	NEIGHBOR_34		chr1	Nan						PROMOTER_24,
			#	NEIGHBOR_35		chr1	ENHANCER_4				Nan
			#	NEIGHBOR_36		chr1	ENHANCER_18,ENHANCER_78	Nan
			#	:

	#-------------
	bin_table = pd.read_csv("MyProject/data/table/region/bin/GM12878_bins.csv", usecols=["id", "start", "end"])
	print(bin_table.head())
	print(f"周辺領域の幅: {bin_length}")
	print(f"周辺領域の数: {len(bin_table)}")

	enhancer_table = pd.read_csv("MyProject/data/table/region/enhancer/GM12878_enhancers.csv", usecols=["id", "start", "end"])
	bin2enhancer_label = [""] * len(bin_table) # key: エンハンサーの番号, value: 周辺領域の番号

	for _, data in tqdm(enhancer_table.iterrows()):
		enhancer_id = data["id"]
		enhancer_start = data["start"]
		enhancer_end = data["end"]

		bin_id = enhancer_start // bin_length
		bin_start = bin_id * bin_length
		bin_end = bin_start + bin_length
		target_bin = -1 # 注目エンハンサーと最も重なっている領域番号の初期化
		now_score = -1 # 注目エンハンサーとどれだけ重なっているかを示す指標
		while bin_start < enhancer_end:
			score = min(bin_end, enhancer_end) - max(bin_start, enhancer_start)
			if now_score < score:
				score = now_score
				target_bin = bin_id
			bin_id += 1
			bin_start += bin_length
			bin_end += bin_length
		bin2enhancer_label[target_bin] += enhancer_id + ","

	del enhancer_table
	
	promoter_table = pd.read_csv("MyProject/data/table/region/promoter/GM12878_promoters.csv", usecols=["id", "start", "end"])
	bin2promoter_label = [""] * len(bin_table) # key: プロモーターの番号, value: 周辺領域の番号

	for _, data in tqdm(promoter_table.iterrows()):
		promoter_id = data["id"]
		promoter_start = data["start"]
		promoter_end = data["end"]
		target_bin = -1 # 注目プロモーターと最も重なっている領域番号の初期化

		bin_id = promoter_start // bin_length
		bin_start = bin_id * bin_length
		bin_end = bin_start + bin_length
		now_score = -1 # 注目プロモーターとどれだけ重なっているかを示す指標
		while bin_start < promoter_end:
			score = min(bin_end, promoter_end) - max(bin_start, promoter_start)
			if now_score < score:
				score = now_score
				target_bin = bin_id
			bin_id += 1
			bin_start += bin_length
			bin_end += bin_length
		bin2promoter_label[target_bin] += promoter_id + ","

	del promoter_table

	print(bin2enhancer_label[:100])
	print(bin2promoter_label[:100])



make_region_label_table("GM12878", 5000)