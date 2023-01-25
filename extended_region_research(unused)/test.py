import pickle
from utils import pickle_load


result_dicts = pickle_load("/Users/ylwrvr/卒論/Koga_code/data/result/GM12878,d=100,way_of_kmer=normal,k=6,s=1,N=1,kmin=-1,kmax=-1,way_of_cv=random,clf=GBRT,trees=4000.pickle")
for key, dic in result_dicts.items():
	print(key)
	print(dic["macro avg"]["f1-score"])