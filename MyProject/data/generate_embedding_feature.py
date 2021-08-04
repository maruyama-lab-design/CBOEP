# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import pandas as pd

cell_line_list = ["GM12878"]
region_type_list = ["bin", "enhancer", "promoter"]
for cl in cell_line_list:
	for region_type in region_type_list:
		print(cl, region_type, "開始")
		csv_path  = "MyProject/data/table/"+region_type+"/"+cl+"_"+region_type+"s.csv"
		df = pd.read_csv(csv_path)
		# 空配列を用意 ここにsentenceを入れていく
		corpus = []
		for index, row in df.iterrows():
			corpus.append(TaggedDocument(row["seq"], row["id"]))


		model = Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=8, epochs=10)
		model.build_vocab(corpus)
		model.train(
			corpus,
			total_examples=model.corpus_count,
			epochs=model.epochs
		)

		# embedding vector 保存
		model.save("MyProject/data/embedding/"+region_type+"/"+cl+"_"+region_type+"s.d2v")