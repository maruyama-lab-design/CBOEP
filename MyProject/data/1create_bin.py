import os
import time

cell_line_list = ["GM12878"]


def create_bin(bin_length):
	# bin_length の長さずつ reference genome から 切り出して fasta 形式で保存

	# reference genome 
	hg19_path = "MyProject/data/fasta/hg19.fa"

	# 染色体毎にどれくらいの長さあるのかを length_by_chr に記録しておく
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

	for cl in cell_line_list:
		print(f"{cl} 開始")

		bin_bed_path = "Myproject/data/bed/bin/"+cl+"_bins.bed"
		bin_fasta_path = "Myproject/data/fasta/bin/"+cl+"_bins.fa"

		# bin の bed file を作る
		print("bedfile を区間長 "+str(bin_length)+" で作る")
		with open(bin_bed_path, "w") as f_bed:
			for chr in range(1, 23):
				start = 0
				end = start + bin_length
				while end < length_by_chr[chr]:
					text = "chr" + str(chr) + "\t" + str(start) + "\t" + str(end) + "\n"
					f_bed.write(text)
					start += bin_length
					end += bin_length

		# bed -> fasta
		print("bed -> fasta 開始")
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ bin_bed_path +" -fo "+ bin_fasta_path)

		# 塩基配列を全て小文字へ
		seqs = ""
		with open(bin_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(bin_fasta_path, "w") as fout:
			fout.write(seqs)

create_bin(5000)