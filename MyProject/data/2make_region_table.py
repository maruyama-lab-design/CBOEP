import pandas as pd
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def seq2sentence(k, stride, seq):
	#-----説明-----
	# seq(塩基配列) を k-mer に区切り、sentence で返す
	# 返り値である sentence 内の k-mer 間は空白区切り
	#-------------

	length = len(seq)
	sentence = ""
	start_pos = 0
	while start_pos <= length - k:
		# k-merに切る
		word = seq[start_pos : start_pos + k]
		
		# 切り出したk-merを書き込む
		sentence += word + ' '

		start_pos += stride

	return sentence


def make_region_table(args):
	# -----説明-----
	# region_type(enhancer, promoter, neighbor のいずれか) を入力とし、
	# 塩基配列(enhancer, promoter, neighbor)についてのテーブルデータを作成します.
	# 前提として、全領域の bedfile, fastafile が存在する必要があります.

		# enhancer のテーブルデータの例
			#	id				chr   	start	end		n_cnt	seq
			#	ENHANCER_34		chr1	235686	235784	0		acgtcdgttcg...

	# -------------
	
	for region_type in args.region_type_list:
		print(f"全ての {region_type} 領域について csvファイルを作成します.")
		# 細胞株毎にループ
		for cell_line in args.cell_line_list:
			print(f"{cell_line} 開始")
			bed_file = open(f"{args.my_data_folder_path}/bed/{region_type}/{cell_line}_{region_type}s.bed", "r")
			fasta_file = open(f"{args.my_data_folder_path}/fasta/{region_type}/{cell_line}_{region_type}s.fa", "r")

			id = 0
			region_ids = [] # ENHANCER_0 などの id を入れていく
			chrs = [] # chr1 などを入れていく
			starts = []	# start pos を入れていく
			ends = [] # end pos を入れていく
			n_cnts = [] # sequence 内の "n" の個数を入れていく

			bed_lines = bed_file.read().splitlines()
			fasta_lines = fasta_file.read().splitlines()

			for fasta_line in fasta_lines:

				# ">chr1:17000-18000" のような行を飛ばす.
				if fasta_line[0] == ">":
					continue

				bed_line_list = bed_lines[id].split("\t")
				chrs.append(bed_line_list[0])
				starts.append(bed_line_list[1])
				ends.append(bed_line_list[2])
				n_cnt = fasta_line.count("n")
				n_cnts.append(n_cnt)
				region_id = region_type.upper() + "_" + str(id)
				region_ids.append(region_id)
				id += 1

			df = pd.DataFrame({
				"id":region_ids,
				"chr":chrs,
				"start":starts,
				"end":ends,
				"n_cnt":n_cnts,
			})
			df.to_csv(f"{args.my_data_folder_path}/table/region/{region_type}/{cell_line}_{region_type}s.csv")

			bed_file.close()
			fasta_file.close()
			print(f"{cell_line} 完了")

		print(f"{region_type} 終了")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='各regionタイプ(enhancer, promoter, neighbor)毎のテーブルデータを作成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	# parser.add_argument("-region_type_list", nargs="+", default=["enhancer", "promoter", "neighbor"])
	parser.add_argument("-region_type_list", nargs="+", default=["neighbor"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", help="neighborの長さ", type=int, default=5000)
	args = parser.parse_args()

	make_region_table(args)

