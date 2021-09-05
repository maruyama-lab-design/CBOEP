import os

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
# 細胞株毎のループは関数の外で
#--------------

cell_line_list = ["GM12878", "K562"]

# numpy matrix (contact map), M

# M >= 0.7 M_boolan 

def create_EnhPrm():
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	hg19_path = "MyProject/data/fasta/hg19.fa"

	# 細胞株毎にループ
	for cl in cell_line_list:
		# input bed
		enhancer_bed_path = "Myproject/data/bed/enhancer/"+cl+"_enhancers.bed"
		promoter_bed_path = "Myproject/data/bed/promoter/"+cl+"_promoters.bed"

		# output fasta
		enhancer_fasta_path = "Myproject/data/fasta/enhancer/"+cl+"_enhancers.fa"
		promoter_fasta_path = "Myproject/data/fasta/promoter/"+cl+"_promoters.fa"

		# bedtools で hg19 を bed 切り出し → fasta に保存
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ enhancer_bed_path +" -fo "+ enhancer_fasta_path)
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ promoter_bed_path +" -fo "+ promoter_fasta_path)

		# 塩基配列を全て小文字へ
		seqs = ""
		with open(enhancer_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(enhancer_fasta_path, "w") as fout:
			fout.write(seqs)
		
		with open(promoter_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(promoter_fasta_path, "w") as fout:
			fout.write(seqs)


create_EnhPrm()