from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import os
import argparse


def tagged_sentence_generator(filename_list):
	for filename in filename_list:
		print(f"{filename}を読み込み中...")
		with open(filename, "r") as fin: # file読み込み
			for line in fin:
				tag_and_sentence = line.split()
				yield TaggedDocument(tag_and_sentence[1:], [tag_and_sentence[0]])


class TaggedSentencesIterator():
	# これは正しく動いてそう　epochごとにシャッフルはしてない？
	# 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
	def __init__(self, generator_function, filename_list):
		self.generator_function = generator_function
		self.filename_list = filename_list
		self.generator = self.generator_function(self.filename_list)

	def __iter__(self):
		# reset the generator
		self.generator = self.generator_function(self.filename_list)
		return self

	def __next__(self):
		result = next(self.generator)
		if result is None:
			raise StopIteration
		else:
			return result


def train_doc2vecModel(args, tagged_sentence_iterator):
	model = Doc2Vec(
		tagged_sentence_iterator,
		min_count=1,
		window=args.window,
		vector_size=args.vector_size,
		sample=1e-4,
		negative=5,
		workers=8,
		epochs=args.epochs
	)
	return model
			

def ep2vec_stage1_v2(args):
	# k_list = "1,2,3" のようになっている．

	input_dir = os.path.join(os.path.dirname(__file__), "ep2vec_preprocess", args.cell_line, args.way_of_kmer)
	input_path = ""
	if args.way_of_kmer == "normal":
		k_list = args.k_list.split(",")
		print(k_list)
		input_path_list = []
		for k in k_list:
			input_path_list.append(os.path.join(input_dir, f"{k}_{args.stride}_concatenated.sent"))
	elif args.way_of_kmer == "random":
		# input_path = os.path.join(input_dir, f"{args.kmin}_{args.kmax}_{args.sentenceCnt}_concatenated.sent")
		print("エラー！！")
	output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_d2v", args.cell_line, args.way_of_kmer)
	os.system(f"mkdir -p {output_dir}")
	if args.way_of_kmer == "normal":
		output_path = os.path.join(output_dir, f"{args.k_list}_{args.stride}.d2v")
	elif args.way_of_kmer == "random":
		# output_path = os.path.join(output_dir, f"{args.kmin}_{args.kmax}_{args.sentenceCnt}.d2v")
		print("エラー！！")

	tagged_sentence_itr = TaggedSentencesIterator(tagged_sentence_generator, input_path_list)
	print("stage1 training...")
	doc2vec = train_doc2vecModel(args, tagged_sentence_itr)
	print("completed")
	doc2vec.save(output_path)
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k_list", help="k-merのk", default="1,2,3,4,5,6")
	parser.add_argument("--stride", type=int, default=1, help="固定帳のk-merの場合のstride")
	parser.add_argument("--kmax", help="k-merのk", type=int, default=6)
	parser.add_argument("--kmin", help="k-merのk", type=int, default=3)
	parser.add_argument("--sentenceCnt", help="何個複製するか", type=int, default=3)
	parser.add_argument("--way_of_kmer", choices=["normal", "random"], default="normal")
	parser.add_argument("--vector_size", help="分散表現の次元", type=int, default=100)
	parser.add_argument("--window", help="doc2vecパラメータ", type=int, default=10)
	parser.add_argument("--epochs", type=int, default=10)
	args = parser.parse_args()

	# cell_line_list = ["GM12878", "HUVEC", "HeLa-S3", "NHEK", "IMR90"]
	k_mer_set = ["1", "2", "3", "4", "5", "6", "1,2,3,4,5,6", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6"]
	cell_line_list = ["GM12878"]
	for cl in cell_line_list:
		for k_mer in k_mer_set:
			args.cell_line = cl
			args.k_list = k_mer
			ep2vec_stage1_v2(args)