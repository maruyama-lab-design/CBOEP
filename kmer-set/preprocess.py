import argparse
import pandas as pd
import urllib.request
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pybedtools
import os
from random import randint

# 塩基配列 ダウンロード
# 塩基配列 切り出し (reverse complement 含め)

def unzip_gzfile(gzfile, output_path):
	with gzip.open(gzfile, "rt") as fin, open(output_path, "w") as fout:
			fout.write(fin.read())


def download_referenceGenome(args):
	output_dir = args.seq_dir
	os.system(f"mkdir -p {output_dir}")

	output_gz_path = os.path.join(output_dir, "hg19.fa.gz")
	if os.path.exists(output_gz_path) == False:
		print("download reference genome gzip...")
		url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz"
		urllib.request.urlretrieve(url, output_gz_path)
		print("downloaded")

	output_referenceGenome_path = os.path.join(output_dir, "hg19.fa")
	if os.path.exists(output_referenceGenome_path) == False:
		print("unzip reference genome...")
		unzip_gzfile(output_gz_path, output_referenceGenome_path)
		print("unzipped")


def download_bedfile(args):
	output_dir = os.path.join(args.seq_dir, args.cell_line)
	os.system(f"mkdir -p {output_dir}")
	
	for regionType in ["enhancer", "promoter"]:
		output_path = os.path.join(output_dir, f"{regionType}s.bed")
		if os.path.exists(output_path):
			continue
		
		print(f"download {args.cell_line} {regionType} bed...")
		targetfinder_root = "https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/"
		url = os.path.join(targetfinder_root, args.cell_line, "output-ep", f"{regionType}s.bed")
		urllib.request.urlretrieve(url, output_path)
		print("downloaded")



def make_fastafile(args):
	input_referenceGenome_path = args.reference_genome

	seq_dir = os.path.join(args.seq_dir, args.cell_line)
	os.system(f"mkdir -p {seq_dir}")

	for regionType in ["enhancer", "promoter"]:
		input_bed_path = os.path.join(seq_dir, f"{regionType}s.bed")
		seq_path = os.path.join(seq_dir, f"{regionType}s.fa")
		if os.path.exists(seq_path):
			continue

		print(f"{args.cell_line} {regionType} make fasta...")
		bed = pybedtools.BedTool(input_bed_path)
		seq = bed.sequence(fi=input_referenceGenome_path, name=True) # ここで切り出しているらしい
		# tmpfileに書き込み（reverse comlement無し）
		with open(seq_path, "w") as fout:
			fout.write(open(seq.seqfn).read())
		print(f"saved in {seq_path}")
		

def make_kmer_words(k, stride, sequence):
	#-----説明-----
	# sequence(塩基配列) を k-mer に区切り、sentence で返す
	# 返り値である sentence は k-mer(word) を tab区切りで並べた string
	#-------------

	sequence = sequence.replace("\n", "") # 改行マークをとる
	length = len(sequence)
	sentence = ""
	start_pos = 0
	while start_pos <= length - k:
		# k-merに切る
		word = sequence[start_pos : start_pos + k]
		
		# 切り出したk-merを書き込む
		sentence += word + "\t"

		start_pos += stride

	return sentence



def make_sentencefile(args):

	for regionType in ["enhancer", "promoter"]:
		input_path = os.path.join(args.seq_dir, args.cell_line, f"{regionType}s.fa")
		seq_dir = os.path.join(args.seq_dir, args.cell_line)
		os.system(f"mkdir -p {seq_dir}")



		for k_stride in args.k_stride_set.split(","):
			[k, stride] = k_stride.split("_")
			k, stride = int(k), int(stride)
			seq_path = os.path.join(seq_dir, f"{k}_{stride}_{regionType}s.sent")
			if os.path.exists(seq_path):
				print(f"{seq_path} is already existed. skipped.")
				continue


			print(f"make {seq_path}...")
			with open(input_path, "r") as fin, open(seq_path, "w") as fout:
				for record in SeqIO.parse(fin, "fasta"):
					# tmpfileから欠損配列を除外と，reverse comp を作る
					seq = str(record.seq).lower()
					if seq.count("n") > 0: # 欠損配列を除外
						continue
					if len(seq) < k: # 短すぎる配列を除外
						continue

					rc_seq = str(record.seq.reverse_complement()).lower()

					fout.write(str(record.id) + "\t" + make_kmer_words(k, stride, seq) + "\n")
					fout.write(str(record.id) + "\t" + make_kmer_words(k, stride, rc_seq) + "\n") # complement 配列
			print(f"saved in {seq_path}")


def concat_EnhPrm_sentfile(args):
	input_dir = os.path.join(args.seq_dir, args.cell_line)
	for k_stride in args.k_stride_set.split(","):
		[k, stride] = k_stride.split("_")
		k, stride = int(k), int(stride)

		enh_path = os.path.join(input_dir, f"{k}_{stride}_enhancers.sent")
		prm_path = os.path.join(input_dir, f"{k}_{stride}_promoters.sent")
		output_path = os.path.join(input_dir, f"{k}_{stride}_concatenated.sent")

		if os.path.exists(output_path):
			print(f"{output_path} is already existed. skipped.")
			continue


		with open(enh_path) as f1, open(prm_path) as f2, open(output_path, "w") as fout:
			fout.write(f1.read())
			fout.write(f2.read())
		print(f"saved in {output_path}")


def preprocess(args):
	download_referenceGenome(args)
	download_bedfile(args)
	make_fastafile(args)

	make_sentencefile(args)
	concat_EnhPrm_sentfile(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--targetfinder_root", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
	parser.add_argument("--ep2vec_root", help="ep2vecのgitのurl", default="https://raw.githubusercontent.com/wanwenzeng/ep2vec/master")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k", help="k-merのk", type=int, default=6)
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--kmax", help="k-merのk", type=int, default=6)
	parser.add_argument("--kmin", help="k-merのk", type=int, default=3)
	parser.add_argument("--sentenceCnt", help="何個複製するか", type=int, default=3)
	parser.add_argument("--way_of_kmer", choices=["normal", "random"], default="normal")
	args = parser.parse_args()

	cell_line_list = ["K562"]
	k_list = [7, 8, 9, 10]
	for cl in cell_line_list:
		for k in k_list:
			args.k = k
			args.cell_line = cl
			ep2vec_preprocess(args)