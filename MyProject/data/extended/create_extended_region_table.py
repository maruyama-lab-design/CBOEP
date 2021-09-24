import pandas as pd
import argparse

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def make_extended_region_table(args):
	# -----説明-----
	# 前提として、全領域の bedfile, fastafile が存在する必要があります.

		# enhancer のテーブルデータの例
			#	id				chr   	start	end		n_cnt	seq
			#	ENHANCER_34		chr1	235686	235784	0		acgtcdgttcg...

	# -------------

	print(f"全ての エンハンサー 領域について csvファイルを作成します.")
	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		print(f"{cell_line} 開始")

		enhancer_fasta_file = open(f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa", "r")

		# enhancer...
		id = 0
		names = [] # chr1:900000-9100000 など
		region_ids = [] # ENHANCER_0 などの id を入れていく
		chrs = [] # chr1 などを入れていく
		starts = []	# start pos を入れていく
		ends = [] # end pos を入れていく
		n_cnts = [] # sequence 内の "n" の個数を入れていく

		fasta_lines = enhancer_fasta_file.readlines()

		for fasta_line in fasta_lines:

			# ">chr1:17000-18000" のような行.
			if fasta_line[0] == ">":
				region_id = "ENHANCER_" + str(id)
				region_ids.append(region_id)

				name = fasta_line[1:].replace("\n", "")
				chr, range_txt = name.split(":")[0], name.split(":")[1]
				start_pos, end_pos = range_txt.split("-")[0], range_txt.split("-")[1]
				names.append(name)
				chrs.append(chr)
				starts.append(start_pos)
				ends.append(end_pos)
			else:
				n_cnt = fasta_line.count("n")
				n_cnts.append(n_cnt)

				id += 1


		df = pd.DataFrame({
			"name":names,
			"id":region_ids,
			"chr":chrs,
			"start":starts,
			"end":ends,
			"n_cnt":n_cnts,
		})
		df.to_csv(f"{args.my_data_folder_path}/table/region/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.csv")

		enhancer_fasta_file.close()
		print(f"{cell_line} 完了")

	
	print(f"全ての プロモーター 領域について csvファイルを作成します.")
	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		print(f"{cell_line} 開始")

		promoter_fasta_file = open(f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa", "r")

		# promoter...
		id = 0
		names = [] # chr1:900000-9100000 など
		region_ids = [] # PROMOTER_0 などの id を入れていく
		chrs = [] # chr1 などを入れていく
		starts = []	# start pos を入れていく
		ends = [] # end pos を入れていく
		n_cnts = [] # sequence 内の "n" の個数を入れていく

		fasta_lines = promoter_fasta_file.readlines()

		for fasta_line in fasta_lines:

			# ">chr1:17000-18000" のような行.
			if fasta_line[0] == ">":
				region_id = "PROMOTER_" + str(id)
				region_ids.append(region_id)

				name = fasta_line[1:].replace("\n", "")
				chr, range_txt = name.split(":")[0], name.split(":")[1]
				start_pos, end_pos = range_txt.split("-")[0], range_txt.split("-")[1]
				names.append(name)
				chrs.append(chr)
				starts.append(start_pos)
				ends.append(end_pos)
			else:
				n_cnt = fasta_line.count("n")
				n_cnts.append(n_cnt)

				id += 1


		df = pd.DataFrame({
			"name":names,
			"id":region_ids,
			"chr":chrs,
			"start":starts,
			"end":ends,
			"n_cnt":n_cnts,
		})
		df.to_csv(f"{args.my_data_folder_path}/table/region/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.csv")

		promoter_fasta_file.close()
		print(f"{cell_line} 完了")

		




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='各regionタイプ(enhancer, promoter, neighbor)毎のテーブルデータを作成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-region_type_list", nargs="+", default=["enhancer", "promoter"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", help="neighborの長さ", type=int, default=5000)
	parser.add_argument("-E_extended_left_length", type=int, default=100)
	parser.add_argument("-E_extended_right_length", type=int, default=100)
	parser.add_argument("-P_extended_left_length", type=int, default=100)
	parser.add_argument("-P_extended_right_length", type=int, default=100)
	args = parser.parse_args()

	make_extended_region_table(args)

