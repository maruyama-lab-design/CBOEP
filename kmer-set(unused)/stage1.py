from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import os
import argparse

import random # シャッフル用


def tagged_sentence_generator(filename_list): # TaggedSentencesIterator内部のgeneratorである．
	for filename in filename_list:
		print(f"{filename}を読み込み中...")
		with open(filename, "r") as fin: # file読み込み
			lines = fin.readlines()
			print(f"シャッフル...")
			random.shuffle(lines)
			lines = [line.strip() for line in lines]
			print("一行目")
			print(lines[0])
			for line in lines:
				tag_and_sentence = line.split() # 空白くぎりでlistへ
				yield TaggedDocument(tag_and_sentence[1:], [tag_and_sentence[0]])


class TaggedSentencesIterator():
	# これは正しく動いてそう　epochごとにシャッフルはしてない？
	# 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
	def __init__(self, generator_function, filename_list):
		self.generator_function = generator_function
		self.filename_list = filename_list
		random.shuffle(self.filename_list)
		self.generator = self.generator_function(self.filename_list)

		print(f"シャッフル後のfileのlistは：")
		print(f"{filename_list}")


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
	print("train doc2vec...")
	model = Doc2Vec(
		tagged_sentence_iterator,
		min_count=1,
		window=args.window, # 10
		vector_size=args.embedding_vector_dimention, # 100
		sample=1e-4,
		negative=5,
		workers=8,
		epochs=args.epochs # 10
	)
	return model
			

def stage1(args):
	# k_list = "1,2,3" のようになっている．

	input_dir = os.path.join(args.seq_dir, args.cell_line)
	k_stride_set = args.k_stride_set.split(",")
	print(f"入力に使うk-mer_stride集合:{k_stride_set}")
	input_path_list = []
	for k_s in k_stride_set:
		[k, stride] = k_s.split("_")
		k, stride = int(k), int(stride)
		input_path_list.append(os.path.join(input_dir, f"{k}_{stride}_concatenated.sent"))

	d2v_dir = os.path.join(args.d2v_dir, args.cell_line, str(args.embedding_vector_dimention))
	os.system(f"mkdir -p {d2v_dir}")

	d2v_path = os.path.join(d2v_dir, f"{args.k_stride_set}.d2v") # 保存先

	tagged_sentence_itr = TaggedSentencesIterator(tagged_sentence_generator, input_path_list)

	doc2vec_model = train_doc2vecModel(args, tagged_sentence_itr)
	doc2vec_model.save(d2v_path)

	print("paragraph vector completed!!")
	print(f"saved in {d2v_path}")
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k_set", help="k-merのk", default="1,2,3,4,5,6")
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
	k_set_list = ["1", "2", "3", "4", "5", "6", "7", "8", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6", "1,2,3,4,5,6,7,8"]
	cell_line_list = ["K562"]
	for cl in cell_line_list:
		for k_set in k_set_list:
			args.cell_line = cl
			args.k_set = k_set
			args.epochs = 10
			ep2vec_stage1_v2(args)