import pandas as pd

cell_line_list = ["GM12878"]

def seq2sentence(k, stride, seq): # k-mer
	# sentence 内の word 間は空白区切り

	length = len(seq)
	sentence = ""
	start_pos = 0
	while start_pos <= length-k:
		# k-merに切る
		word = seq[start_pos : start_pos + k]
		
		# 切り出したk-merを書き込む
		sentence += word + ' '

		start_pos += stride

	return sentence


def make_region_table(region_type):
	# 入力に用いる塩基配列(enhancer, promoter, bin)についてのテーブルデータを作成します.
	# 前提として、全領域の bedfile, fastafile が存在する必要があります.
	# enhancer, promoter のテーブルデータに関する、行データの例
		#	id				chr   	start	end		n_cnt	seq   			sentence
		#	ENHANCER_34		chr1	235686	235784	0		acgtcdgttcg...	acg cgt gtc tcd cdg ...
	
	region_type = region_type.lower()
	region_types = ["enhancer", "promoter", "bin"]
	assert region_type in region_types, "enhancer, promoter, bin のいずれかを選択してください"

	print(f"全ての {region_type} 領域について csvファイルを作成します.")
	for cl in cell_line_list:
		print(f"{cl} 開始")
		bed_file = open("MyProject/data/bed/"+region_type+"/"+cl+"_"+region_type+"s.bed", "r")
		fasta_file = open("MyProject/data/fasta/"+region_type+"/"+cl+"_"+region_type+"s.fa", "r")

		id = 0
		region_ids = [] # ENHANCER_0 などの id を入れていく
		chrs = [] # chr1 などを入れていく
		starts = []	# start pos を入れていく
		ends = [] # end pos を入れていく
		seqs = [] # 塩基配列 の sequence を入れていく
		n_cnts = [] # sequence 内の "n" の個数を入れていく
		# sentences = [] # 上の sequence を k-mer に区切ったものを入れていく.

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
			seqs.append(fasta_line)
			# sentence = seq2sentence(6, 1, line)
			# sentences.append(sentence)
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
			"seq":seqs,
			# "sentence":sentences,
		})
		df.to_csv("MyProject/data/table/"+region_type+"/"+cl+"_"+region_type+"s.csv")

		bed_file.close()
		fasta_file.close()
		print(f"{cl} 完了")

	print(f"{region_type} 終了")

make_region_table("enhancer")
make_region_table("promoter")
# make_region_table("bin")

