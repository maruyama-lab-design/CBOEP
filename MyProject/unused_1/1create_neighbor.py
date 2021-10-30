import os
import time
import argparse

def create_neighbor(args):
	# ----- 説明 -----
	# neighbor_length の長さずつ reference genome から 切り出して fasta 形式で保存
	# ---------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# 染色体毎の長さを length_by_chr に記録しておく
	with open(f"{args.my_data_folder_path}/chrome_sizes.txt", "r") as f_size:
		length_by_chr = {} # key: 染色体番号, value: 長さ
		lines = f_size.read().splitlines()
		for line in lines:
			chr = line.split("\t")[0]
			length = int(line.split("\t")[1])
			length_by_chr[chr] = length

	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		print(f"{cell_line} 開始")

		neighbor_bed_path = f"{args.my_data_folder_path}/bed/neighbor/{cell_line}_neighbors.bed" # input
		neighbor_fasta_path = f"{args.my_data_folder_path}/fasta/neighbor/{cell_line}_neighbors.fa" # output

		# neighbor の bed file を作る
		print("bedfile を区間長 "+str(args.neighbor_length)+" で作ります")
		with open(neighbor_bed_path, "w") as f_bed:
			for i in range(1, 23):
				chr = "chr" + str(i)
				start = 0
				end = start + args.neighbor_length
				while end < length_by_chr[chr]:
					text = "chr" + str(i) + "\t" + str(start) + "\t" + str(end) + "\n"
					f_bed.write(text)
					start += args.neighbor_length
					end += args.neighbor_length
			for i in ["X", "Y"]:
				chr = "chr" + str(i)
				start = 0
				end = start + args.neighbor_length
				while end < length_by_chr[chr]:
					text = "chr" + i + "\t" + str(start) + "\t" + str(end) + "\n"
					f_bed.write(text)
					start += args.neighbor_length
					end += args.neighbor_length

		# bed -> fasta
		print("bed -> fasta 開始")
		os.system("bedtools getfasta -fi "+ reference_genome_path +" -bed "+ neighbor_bed_path +" -fo "+ neighbor_fasta_path)
		print(f"{neighbor_fasta_path} に保存完了")

		# 塩基配列を全て小文字へ
		print("塩基配列を小文字にします.")
		seqs = ""
		with open(neighbor_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(neighbor_fasta_path, "w") as fout:
			fout.write(seqs)

		print(f"{cell_line} 終了")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='reference genome全体を先頭から固定長の長さずつに切り出し、周辺領域(neighbor)とします.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", help="neighborの長さ", type=int, default=5000)
	args = parser.parse_args()

	create_neighbor(args)