import pandas as pd
from tqdm import tqdm
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
# まだ未完成
#--------------


def test_make_region_label_table(args):
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
        enhancer_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers.csv", usecols=["id", "chr", "start", "end"])
        print(enhancer_table.head())
        print(f"エンハンサーの数: {len(enhancer_table)}")
        promoter_table = pd.read_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters.csv", usecols=["id", "chr", "start", "end"])
        print(promoter_table.head())
        print(f"プロモーターの数: {len(promoter_table)}")

        neighbor2enhancer_label = [""] * len(neighbor_table) # index: 周辺領域の番号, value: エンハンサーのid
        for (n_chr, neighbor_table_by_chr), (e_chr, enhancer_table_by_chr) in zip(neighbor_table.groupby("chr"), enhancer_table.groupby("chr")):
            if n_chr != e_chr:
                print("エラー!!!")
                exit()
            chr = n_chr
            print(f"{chr}...")
            # print(neighbor_table_by_chr.head())
            # print(enhancer_table_by_chr.head())

            for _, data in tqdm(enhancer_table_by_chr.iterrows()):
                enhancer_id = data["id"]
                enhancer_start = data["start"]
                enhancer_end = data["end"]
                enhancer_center = (enhancer_start + enhancer_end) // 2

                target_neighbor_index = enhancer_center // args.neighbor_length
                target_neighbor_id = neighbor_table_by_chr.iloc[target_neighbor_index]["id"]
                neighbor2enhancer_label[int(target_neighbor_id.split("_")[1])] += enhancer_id + ","

        del enhancer_table
        print(neighbor2enhancer_label[:100])

        neighbor2promoter_label = [""] * len(neighbor_table) # index: 周辺領域の番号, value: プロモーターのid
        for (n_chr, neighbor_table_by_chr), (p_chr, promoter_table_by_chr) in zip(neighbor_table.groupby("chr"), promoter_table.groupby("chr")):
            if n_chr != p_chr:
                print("エラー!!!")
                exit()
            chr = n_chr
            print(f"{chr}...")
            # print(neighbor_table_by_chr.head())
            # print(promoter_table_by_chr.head())

            for _, data in tqdm(promoter_table_by_chr.iterrows()):
                promoter_id = data["id"]
                promoter_start = data["start"]
                promoter_end = data["end"]
                promoter_center = (promoter_start + promoter_end) // 2

                target_neighbor_index = promoter_center // args.neighbor_length
                target_neighbor_id = neighbor_table_by_chr.iloc[target_neighbor_index]["id"]
                neighbor2promoter_label[int(target_neighbor_id.split("_")[1])] += promoter_id + ","

        del promoter_table
        print(neighbor2promoter_label[:100])
        
        new_df = pd.DataFrame({ 'neighbor_id' : list(neighbor_table["id"]),
                                'enhancer_id' : neighbor2enhancer_label,
                                'promoter_id' : neighbor2promoter_label,})
        
        new_df.to_csv("MyProject/test/table/test.csv")            

        # for index, (enhancer_label, promoter_label) in enumerate(zip(neighbor2enhancer_label, neighbor2promoter_label))
            

	    

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='各regionタイプ(enhancer, promoter, neighbor)毎のテーブルデータを作成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-region_type_list", nargs="+", default=["enhancer", "promoter", "neighbor"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", help="neighborの長さ", type=int, default=5000)
	args = parser.parse_args()

	test_make_region_label_table(args)