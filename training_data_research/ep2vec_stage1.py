from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import os
import argparse


def tagged_sentence_generator(filename):
	with open(filename, "r") as fin: # file読み込み
		for line in fin:
			tag_and_sentence = line.split()
			yield TaggedDocument(tag_and_sentence[1:], [tag_and_sentence[0]])


class TaggedSentencesIterator():
	# これは正しく動いてそう　epochごとにシャッフルはしてない？
	# 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
	def __init__(self, generator_function, filename):
		self.generator_function = generator_function
		self.filename = filename
		self.generator = self.generator_function(self.filename)

	def __iter__(self):
		# reset the generator
		self.generator = self.generator_function(self.filename)
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
			

def ep2vec_stage1(args):
	input_path = os.path.join(os.path.dirname(__file__), "ep2vec_preprocess", args.cell_line, "concatenated.sent")
	output_dir = os.path.join(os.path.dirname(__file__), "ep2vec_d2v", args.cell_line)
	os.system(f"mkdir -p {output_dir}")
	output_path = os.path.join(output_dir, "d2v")

	tagged_sentence_itr = TaggedSentencesIterator(tagged_sentence_generator, input_path)
	print("stage1 training...")
	doc2vec = train_doc2vecModel(args, tagged_sentence_itr)
	print("completed")
	doc2vec.save(output_path)
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--dataset", help="どのデータセットを使うか", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--vector_size", help="分散表現の次元", type=int, default=100)
	parser.add_argument("--window", help="doc2vecパラメータ", type=int, default=10)
	parser.add_argument("--epochs", type=int, default=10)
	args = parser.parse_args()

	ep2vec_stage1(args)