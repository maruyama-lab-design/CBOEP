import os

cell_line_list = ["GM12878", "K562"]
# region_types = ["enhancer", "promoter"]

# numpy matrix (contact map), M

# M >= 0.7 M_boolan 

def create_EnhPrm():
	# bedfile を参照し、enhancer, promoter seq の切り出し & fasta形式で保存

	hg19_path = "MyProject/data/fasta/hg19.fa"
	for cl in cell_line_list:
		enhancer_bed_path = "Myproject/data/bed/enhancer/HANCER_"+cl+"_enhancers.bed"
		promoter_bed_path = "Myproject/data/bed/promoter/"+cl+"_promoters.bed"

		enhancer_output_path = "Myproject/data/fasta/enhancer/"+cl+"_enhancers.fa"
		promoter_output_path = "Myproject/data/fasta/promoter/"+cl+"_promoters.fa"

		# bedtools で hg19 を bed 切り出し → fasta に保存
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ enhancer_bed_path +" -fo "+ enhancer_output_path)
		os.system("bedtools getfasta -fi "+ hg19_path +" -bed "+ promoter_bed_path +" -fo "+ promoter_output_path)

		# 塩基配列を全て小文字へ
		with open(enhancer_output_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(enhancer_output_path, "w") as fout:
			fout.write(seqs)
		
		with open(promoter_output_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(promoter_output_path, "w") as fout:
			fout.write(seqs)


create_EnhPrm()