# system modules
import os
import time
import sys
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# random shuffle
from random import shuffle
# numpy
import numpy as np
# pandas
import pandas as pd
# classifier
# from sklearn.cross_validation import StratifiedKFold, cross_val_score # <= 現pythonでは使用できない
from sklearn.model_selection import StratifiedKFold, cross_validate # <= 代わり
from sklearn.ensemble import GradientBoostingClassifier

import argparse

parser = argparse.ArgumentParser(description="エンハンサー, プロモーターの両端を延長したものに対し, doc2vecを行い,EPIs予測モデルの学習, 評価をする.")
parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
parser.add_argument("-my_data_folder_path", help="データのルートとなるフォルダパス")
parser.add_argument("-neighbor_length", default=5000)
parser.add_argument("-embedding_vector_dimention", type=int, default=100)
parser.add_argument("-k", type=int, default=6)
parser.add_argument("-stride", type=int, default=1)
args = parser.parse_args()

enhancers_num = 0
promoters_num = 0
positive_num  = 0
negative_num  = 0

kmer = args.k # the length of k-mer
swin = args.stride # the length of stride
vlen = args.embedding_vector_dimention # the dimension of embedding vector
cl   = "GM12878"      # the interested cell line

#bed2sent: convert the enhancers.bed and promoters.bed to kmer sentense
#bed format: chr start end name
def bed2sent(filename,k,win):
	# if os.path.isfile(f"{args.my_data_folder_path}/fasta/{filename}/{cl}_{filename}s.fa") == False: # エンハンサー, プロモーターの切り出し
	# 	os.system("bedtools getfasta -fi /home/openness/common/igenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa -bed "+filename+".bed -fo "+filename+".fa")
	# 	time.sleep(30)	
	fin   = open(f"{args.my_data_folder_path}/fasta/{filename}/{cl}_{filename}s.fa")
	fout  = open(filename+'s_'+str(k)+'_'+str(swin)+'.sent','w')
	for line in fin:
		if line[0] =='>':
			continue
		else:
			line   = line.strip().lower()
			length = len(line)
			i = 0
			while i<= length-k:
				fout.write(line[i:i+k]+' ')
				i = i + win
			fout.write('\n')
	
#generateTraining: extract the training set from pairs.csv and output the training pair with sentence
def generateTraining():
	global enhancers_num,promoters_num,positive_num,negative_num
	fin1 = open(f"{args.my_data_folder_path}/bed/enhancer/{cl}_enhancers.bed",'r')
	fin2 = open(f"{args.my_data_folder_path}/bed/promoter/{cl}_promoters.bed",'r')
	enhancers = []
	promoters = []
	for line in fin1:
		data = line.strip().split()
		enhancers.append(data[3])
		enhancers_num = enhancers_num + 1
	for line in fin2:
		data = line.strip().split()
		promoters.append(data[3])
		promoters_num = promoters_num + 1
	fin3 = open(f"{args.my_data_folder_path}/train/{cl}_train.csv",'r')
	fout = open('training.txt','w')
	for line in fin3:
		if line[0] == 'b':
			continue
		else:
			data = line.strip().split(',')
			enhancer_index = enhancers.index(data[5])
			promoter_index = promoters.index(data[10])
			fout.write(str(enhancer_index)+'\t'+str(promoter_index)+'\t'+data[7]+'\n')
			if data[7] == '1':
				positive_num = positive_num + 1
			else:
				negative_num = negative_num + 1

# convert the sentence to doc2vec's tagged sentence
class TaggedLineSentence(object):
	def __init__(self, sources):
		self.sources = sources

		flipped = {}

		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.open(source) as fin:
				for item_no, line in enumerate(fin):
					yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

	def to_array(self):
		self.sentences = []
		for source, prefix in self.sources.items():
			with utils.open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences

	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

#train the embedding vector from the file
def doc2vec(name,k,vlen):
	filename  = name+'s_'+str(k)+'_'+str(swin)+'.sent'
	indexname = name.upper()
	sources = {filename:indexname}
	sentences = TaggedLineSentence(sources)
	model = Doc2Vec(min_count=1, window=10, vector_size=vlen, sample=1e-4, negative=5, workers=8)
	model.build_vocab(sentences.to_array())
	model.train(
		sentences.sentences,
		total_examples=model.corpus_count,
		epochs=model.epochs
	)
	# for epoch in range(10):
	# 		model.train(sentences.sentences_perm())
	model.save(name+'s_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')

#train the model and print the result
def train(k,vlen):
	global enhancers_num,promoters_num,positive_num,negative_num
	enhancer_model = Doc2Vec.load('enhancers_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	promoter_model = Doc2Vec.load('promoters_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	arrays = np.zeros((positive_num+negative_num, vlen*2))
	labels = np.zeros(positive_num+negative_num)
	# num    = positive_num+negative_num
	estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	fin = open('training.txt','r')
	i = 0
	for line in fin:
		data = line.strip().split()
		prefix_enhancer = 'ENHANCER_' + data[0]
		prefix_promoter = 'PROMOTER_' + data[1]
		enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
		promoter_vec = promoter_model.docvecs[prefix_promoter]
		enhancer_vec = enhancer_vec.reshape((1,vlen))
		promoter_vec = promoter_vec.reshape((1,vlen))
		arrays[i] = np.column_stack((enhancer_vec,promoter_vec))
		labels[i] = int(data[2])
		i = i + 1

	# 評価する指標
	score_funcs = ['f1', 'roc_auc', 'average_precision']
	# cv = StratifiedKFold(y = labels, n_folds = 10, shuffle = True, random_state = 0)
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	# scores = cross_val_score(estimator, arrays, labels, scoring = 'f1', cv = cv, n_jobs = -1)
	scores = cross_validate(estimator, arrays, labels, scoring = score_funcs, cv = cv, n_jobs = -1)

	# print('f1:')
	# print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	# scores = cross_val_score(estimator, arrays, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
	# print('auc:')
	# print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	# scores = cross_val_score(estimator, arrays, labels, scoring = 'average_precision', cv = cv, n_jobs = -1)
	# print('auc:')
	# print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

	print('F1:', scores['test_f1'].mean())
	print('auROC:', scores['test_roc_auc'].mean())
	print('auPRC:', scores['test_average_precision'].mean())
	f1 = scores['test_f1']
	f1 = np.append(f1, scores['test_f1'].mean())
	auROC = scores['test_roc_auc']
	auROC = np.append(auROC, scores['test_roc_auc'].mean())
	auPRC =  scores['test_average_precision']
	auPRC = np.append(auPRC, scores['test_average_precision'].mean())
	result = pd.DataFrame(
		{
		"F1": f1,
		"auROC": auROC,
		"auPRC": auPRC,
		},
		index = ["1-fold", "2-fold", "3-fold", "4-fold", "5-fold", "6-fold", "7-fold", "8-fold", "9-fold", "10-fold", "mean"]	
	)
	result.to_csv(f"{args.my_data_folder_path}/result/{cl}_ep2vec.csv")

if __name__ == "__main__":
	enhancers_num = 0
	promoters_num = 0
	positive_num  = 0
	negative_num  = 0

	kmer = args.k # the length of k-mer
	swin = args.stride # the length of stride
	vlen = args.embedding_vector_dimention # the dimension of embedding vector
	cl   = "GM12878"      # the interested cell line

	print("k:",kmer,"stride:",swin,"embedding vector size:",vlen, "cell line", cl)

	# bed2sent("promoter",kmer,swin)
	# bed2sent("enhancer",kmer,swin)
	# print('pre process done!')
	# generateTraining()
	# print('generate training set done!')
	# doc2vec("promoter",kmer,vlen)
	# doc2vec("enhancer",kmer,vlen)
	# print('doc2vec done!')
	generateTraining()
	train(kmer,vlen)