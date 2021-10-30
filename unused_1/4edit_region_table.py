import argparse
import pandas as pd

def edit_region_table(args):

	for cell_line in args.cell_line_list:
		enhancer_label_table = pd.read_csv(f"{args.my_data_folder_path}/table/label/enhancer/{cell_line}_enhancers.csv", index_col=0)
		promoter_label_table = pd.read_csv(f"{args.my_data_folder_path}/table/label/promoter/{cell_line}_promoters.csv", index_col=0)
		print(enhancer_label_table.head())
		print(promoter_label_table.head())

		neighbor_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/neighbor/{cell_line}_neighbors.csv", index_col=0)
		print(neighbor_table.head())

		editted_neighbor_index_dict = {} # 削除しないneighborのindex値を入れる.

		for _, data in enhancer_label_table.iterrows():
			neighbor_id = data["neighbor_id"]
			neighbor_index = int(neighbor_id.split("_")[1])
			for i in range(neighbor_index - args.neighbor_cnt, neighbor_index + args.neighbor_cnt + 1):
				if i < 0:
					continue
				if i >= len(neighbor_table):
					continue
				if neighbor_table.loc[neighbor_index, "n_cnt"] > 0:
					continue
				editted_neighbor_index_dict[neighbor_index] = 1
		
		for _, data in promoter_label_table.iterrows():
			neighbor_id = data["neighbor_id"]
			neighbor_index = int(neighbor_id.split("_")[1])
			for i in range(neighbor_index - args.neighbor_cnt, neighbor_index + args.neighbor_cnt + 1):
				if i < 0:
					continue
				if i >= len(neighbor_table):
					continue
				if neighbor_table.loc[neighbor_index, "n_cnt"] > 0:
					continue
				editted_neighbor_index_dict[neighbor_index] = 1
		
		editted_neighbor_index_list = list(editted_neighbor_index_dict.keys())
		sorted(editted_neighbor_index_list)
		print(f"残った領域は {len(editted_neighbor_index_list)} 個")
		editted_neighbor_table = neighbor_table.iloc[editted_neighbor_index_list, :]
		editted_neighbor_table.to_csv(f"{args.my_data_folder_path}/table/region/neighbor/{cell_line}_editted_neighbors.csv", index=False)





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='neighborのテーブルデータから使わない領域を削除します. ここで初めてneighborと呼べるようになる.')
	parser.add_argument("-neighbor_cnt", help="neighborを一つのenhancer/promoterに対し前後いくつまで許容するか", type=int, default=2)
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	args = parser.parse_args()

	edit_region_table(args)