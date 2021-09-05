import os
import time

# メモ ---------
# cell line 毎にループを回すのは関数の外で
# argparse を入れて変数、pathを管理した方が良い
#--------------


cell_line_list = ["GM12878"]
def create_neighbor(neighbor_length):
	# ----- 説明 -----
	# neighbor_length の長さずつ reference genome から 切り出して fasta 形式で保存
	# ---------------

	# reference genome 
	hg19_path = "MyProject/data/fasta/hg19.fa"

	# 染色体毎にどれくらいの長さあるのかを length_by_chr に記録しておく
	# いらない処理なので変更予定あり txtファイルなどで管理して参照した方がわかりやすくて速い
	print("染色体毎に長さを記録")
	with open(hg19_path, "r") as fin:
		length_by_chr = {} # key: 染色体番号, value: 長さ
		now_chr = 0
		lines = fin.read().splitlines()
		seq = ""
		for line in lines:
			if len(line) == 0:
				continue
			if line[0] == ">":
				if now_chr != 0:
					length_by_chr[now_chr] = len(seq)
					# print(len(chromosome_seqs[now_chr]))
				now_chr += 1
				seq = ""
				continue
			seq += line

	# 細胞株毎にループ
	# 関数の外でやった方が良い
	for cl in cell_line_list:
		print(f"{cl} 開始")

		neighbor_bed_path = "Myproject/data/bed/neighbor/"+cl+"_neighbors.bed" # input
		neighbor_fasta_path = "Myproject/data/fasta/neighbor/"+cl+"_neighbors.fa" # output

		# neighbor の bed file を作る
		print("bedfile を区間長 "+str(neighbor_length)+" で作る")
		with open(neighbor_bed_path, "w") as f_bed:
			for chr in range(1, 23):
				start = 0
				end = start + neighbor_length
				while end < length_by_chr[chr]:
					text = "chr" + str(chr) + "\t" + str(start) + "\t" + str(end) + "\n"
					f_bed.write(text)
					start += neighbor_length
					end += neighbor_length

		# bed -> fasta
		print("bed -> fasta 開始")
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ neighbor_bed_path +" -fo "+ neighbor_fasta_path)

		# 塩基配列を全て小文字へ
		seqs = ""
		with open(neighbor_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(neighbor_fasta_path, "w") as fout:
			fout.write(seqs)

create_neighbor(5000)