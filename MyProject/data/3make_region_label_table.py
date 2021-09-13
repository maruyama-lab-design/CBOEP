import pandas as pd
from tqdm import tqdm
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
# まだ未完成
#--------------


def make_region_label_table(args):
	#-----説明-----
	# enhancer, promoter それぞれが、どの neighbor 領域に含まれているかを表すテーブルデータを作成する.
	# 2make_region_table.py で作られるテーブルデータが必要

		# テーブルデータの例
			#	neighbor_id		chr   	enhancer				promoter
			#	:
			#	NEIGHBOR_34		chr1	Nan						PROMOTER_24,
			#	NEIGHBOR_35		chr1	ENHANCER_4				Nan
			#	NEIGHBOR_36		chr1	ENHANCER_18,ENHANCER_78	Nan
			#	:

	#-------------

	for cell_line in args.cell_line_list:
		neighbor_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/neighbor/{cell_line}_neighbors.csv", usecols=["id", "chr", "start", "end"])
		print(neighbor_table.head())
		print(f"周辺領域の幅: {args.neighbor_length}")
		print(f"周辺領域の数: {len(neighbor_table)}")

		# enhancer...
		enhancer_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers.csv", usecols=["id", "chr", "start", "end"])
		print(enhancer_table.head())
		print(f"エンハンサーの数: {len(enhancer_table)}")

		neighbor2enhancer_label = [""] * len(neighbor_table) # index: 周辺領域のindex, value: エンハンサーのid
		enhancer2neighbor_label = [""] * len(enhancer_table) # index: エンハンサーのindex, value: 周辺領域のid
		# 染色体番号毎にループ
		for (n_chr, neighbor_table_by_chr), (e_chr, enhancer_table_by_chr) in zip(neighbor_table.groupby("chr"), enhancer_table.groupby("chr")):
			if n_chr != e_chr:
				print("エラー!!!")
				exit()
			chr = n_chr

			for _, data in tqdm(enhancer_table_by_chr.iterrows(), total=len(enhancer_table_by_chr), desc=f"[{chr}]"):

				enhancer_id = data["id"]
				enhancer_start = data["start"]
				enhancer_end = data["end"]
				enhancer_center = (enhancer_start + enhancer_end) // 2

				target_neighbor_index = enhancer_center // args.neighbor_length
				target_neighbor_id = neighbor_table_by_chr.iloc[target_neighbor_index]["id"]
				neighbor2enhancer_label[int(target_neighbor_id.split("_")[1])] += enhancer_id + ","
				enhancer2neighbor_label[int(enhancer_id.split("_")[1])] = target_neighbor_id


		enhancer_label_table = pd.DataFrame({ 'enhancer_id' : list(enhancer_table["id"]),
								'neighbor_id' : enhancer2neighbor_label,})
		enhancer_label_table.to_csv(f"{args.my_data_folder_path}/table/label/enhancer/{cell_line}_enhancers.csv") 

		del enhancer_table
		del enhancer_label_table

		
		# promoter...
		promoter_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters.csv", usecols=["id", "chr", "start", "end"])
		print(promoter_table.head())
		print(f"プロモーターの数: {len(promoter_table)}")

		neighbor2promoter_label = [""] * len(neighbor_table) # index: 周辺領域のindex, value: プロモーターのid
		promoter2neighbor_label = [""] * len(promoter_table) # index: プロモーターのindex, value: 周辺領域のid
		for (n_chr, neighbor_table_by_chr), (p_chr, promoter_table_by_chr) in zip(neighbor_table.groupby("chr"), promoter_table.groupby("chr")):
			if n_chr != p_chr:
				print("エラー!!!")
				exit()
			chr = n_chr

			for _, data in tqdm(promoter_table_by_chr.iterrows(), total=len(promoter_table_by_chr), desc=f"[{chr}]"):
				promoter_id = data["id"]
				promoter_start = data["start"]
				promoter_end = data["end"]
				promoter_center = (promoter_start + promoter_end) // 2

				target_neighbor_index = promoter_center // args.neighbor_length
				target_neighbor_id = neighbor_table_by_chr.iloc[target_neighbor_index]["id"]
				neighbor2promoter_label[int(target_neighbor_id.split("_")[1])] += promoter_id + ","
				promoter2neighbor_label[int(promoter_id.split("_")[1])] = target_neighbor_id


		promoter_label_table = pd.DataFrame({ 'promoter_id' : list(promoter_table["id"]),
								'neighbor_id' : promoter2neighbor_label,})
		promoter_label_table.to_csv(f"{args.my_data_folder_path}/table/label/promoter/{cell_line}_promoters.csv") 
								
		del promoter_table
		del promoter_label_table
		
		neighbor_label_table = pd.DataFrame({ 'neighbor_id' : list(neighbor_table["id"]),
								'enhancer_id' : neighbor2enhancer_label,
								'promoter_id' : neighbor2promoter_label,})
		neighbor_label_table.to_csv(f"{args.my_data_folder_path}/table/label/neighbor/{cell_line}_neighbors.csv")            


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='各regionタイプ(enhancer, promoter, neighbor)毎のテーブルデータを作成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-region_type_list", nargs="+", default=["enhancer", "promoter", "neighbor"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", help="neighborの長さ", type=int, default=5000)
	args = parser.parse_args()

	make_region_label_table(args)