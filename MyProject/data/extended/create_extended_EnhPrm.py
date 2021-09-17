from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd

import os
import argparse

# メモ ---------
# argparse を入れて変数、pathを管理した方が良い
#--------------

def create_extended_EnhPrm(args):
	# -----説明-----
	# bedファイル を参照し、enhancer, promoter 塩基配列 の切り出し & fasta形式で保存
	# -------------

	# reference genome
	reference_genome_path = f"{args.my_data_folder_path}/reference_genome/hg19.fa"

	# 細胞株毎にループ
	for cell_line in args.cell_line_list:
		# input bed
		extended_enhancer_bed_path = f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.bed"
		extended_promoter_bed_path = f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.bed"

		# output fasta
		extended_enhancer_fasta_path = f"{args.my_data_folder_path}/fasta/enhancer/{cell_line}_enhancers_{args.E_extended_left_length}_{args.E_extended_right_length}.fa"
		extended_promoter_fasta_path = f"{args.my_data_folder_path}/fasta/promoter/{cell_line}_promoters_{args.P_extended_left_length}_{args.P_extended_right_length}.fa"

		if not os.path.exists(extended_enhancer_bed_path):
			print("与えられたextendedエンハンサーのbedfileがありません")
			print("オリジナルのエンハンサーのbedfileから作成します...")
			text = ""
			with open(f"{args.my_data_folder_path}/bed/enhancer/{cell_line}_enhancers.bed", "r") as origin_bed:
				lines = origin_bed.readlines()
				for line in lines:
					line = line.split("\t")
					chr, start_pos, end_pos = line[0], int(line[1]), int(line[2])
					start_pos -= args.E_extended_left_length
					end_pos += args.E_extended_right_length
					name = cell_line + "|" + chr + ":" + str(start_pos) + "-" + str(end_pos)
					text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n"
			with open(extended_enhancer_bed_path, "w") as extended_bed:
				extended_bed.write(text)
		if not os.path.exists(extended_promoter_bed_path):
			print("与えられたextendedプロモーターのbedfileがありません")
			print("オリジナルのプロモーターのbedfileから作成します...")
			text = ""
			with open(f"{args.my_data_folder_path}/bed/promoter/{cell_line}_promoters.bed", "r") as origin_bed:
				lines = origin_bed.readlines()
				for line in lines:
					line = line.split("\t")
					chr, start_pos, end_pos = line[0], int(line[1]), int(line[2])
					start_pos -= args.P_extended_left_length
					end_pos += args.P_extended_right_length
					name = cell_line + "|" + chr + ":" + str(start_pos) + "-" + str(end_pos)
					text += chr + "\t" + str(start_pos) + "\t" + str(end_pos) + "\t" + name + "\n"
			with open(extended_promoter_bed_path, "w") as extended_bed:
				extended_bed.write(text)


		# bedtools で hg19 を bed 切り出し → fasta に保存
		print("bedfileからfastafileを作ります")
		os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_enhancer_bed_path +" -fo "+ extended_enhancer_fasta_path)
		os.system(f"bedtools getfasta -fi {reference_genome_path} -bed "+ extended_promoter_bed_path +" -fo "+ extended_promoter_fasta_path)

		# 塩基配列を全て小文字へ
		seqs = ""
		with open(extended_enhancer_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(extended_enhancer_fasta_path, "w") as fout:
			fout.write(seqs)
		
		with open(extended_promoter_fasta_path, "r") as fout:
			seqs = fout.read()
		seqs = seqs.replace("A", "a").replace("G", "g").replace("C", "c").replace("T", "t").replace("N", "n")
		with open(extended_promoter_fasta_path, "w") as fout:
			fout.write(seqs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='enhancerとpromoterのbedfileとreference genomeのfastafileからenhancerとpromoterのfastafileを生成します.')
	parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
	parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
	parser.add_argument("-neighbor_length", default=5000)
	parser.add_argument("-E_extended_left_length", type=int, default=100)
	parser.add_argument("-E_extended_right_length", type=int, default=100)
	parser.add_argument("-P_extended_left_length", type=int, default=100)
	parser.add_argument("-P_extended_right_length", type=int, default=100)
	args = parser.parse_args()

	create_extended_EnhPrm(args)