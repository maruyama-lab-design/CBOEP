import os
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def create_EnhPrm(args):
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		# input bed
		enhancer_bed_path = f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed"
		promoter_bed_path = f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed"

		# output fasta
		enhancer_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers.fa"
		promoter_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters.fa"

		# bedtools で hg19 を bed 切り出し → fasta に保存
		os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ enhancer_bed_path +" -fo "+ enhancer_fasta_path)
		os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ promoter_bed_path +" -fo "+ promoter_fasta_path)

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='enhancerとpromoterのbedfileとreference genomeのfastafileからenhancerとpromoterのfastafileを生成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	args = parser.parse_args()

	create_EnhPrm(args)