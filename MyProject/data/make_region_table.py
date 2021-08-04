import pandas as pd

cell_line_list = ["GM12878"]

def seq2sentence(k, stride, seq): # k-mer
	# sentence 内の word 間は空白区切り

	length = len(seq)
	sentence = ""
	i = 0
	while i <= length-k:
		# k-merに切る
		word = seq[i : i+k]
		
		# 切り出したk-merを書き込む
		sentence += word + ' '
		i = i + stride
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
		region_ids = []
		chrs = []
		starts = []
		ends = []
		seqs = []
		sentences = []
		n_cnts = []
		bed_lines = bed_file.read().splitlines()
		fasta_lines = fasta_file.read().splitlines()
		for line in fasta_lines:
			if line[0] == ">":
				continue
			bed_line_list = bed_lines[id].split("\t")
			chrs.append(bed_line_list[0])
			starts.append(bed_line_list[1])
			ends.append(bed_line_list[2])
			seqs.append(line)
			sentence = seq2sentence(6, 1, line)
			sentences.append(sentence)
			n_cnt = line.count("n")
			n_cnts.append(n_cnt)
			prefix = region_type.upper() + "_"
			region_ids.append(prefix + str(id))
			id += 1
		df = pd.DataFrame({
			"id":region_ids,
			"chr":chrs,
			"start":starts,
			"end":ends,
			"n_cnt":n_cnts,
			"seq":seqs,
			"sentence":sentences,
		})
		df.to_csv("MyProject/data/table/"+region_type+"/"+cl+"_"+region_type+"s.csv")
		bed_file.close()
		fasta_file.close()
		print(f"{cl} 完了")

	print(f"{region_type} 終了")

make_region_table("enhancer")
make_region_table("promoter")
# make_region_table("bin")

