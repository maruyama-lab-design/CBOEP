from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
import itertools

import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# classifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from t_sne import t_SNE

import data_download


def test():
	d2v_1 = os.path.join("/Users/ylwrvr/卒論/Koga_code/data/d2v/K562,el=0,er=0,pl=0,pr=10000,kmer=random,N=1.d2v")
	d2v_5 = os.path.join("/Users/ylwrvr/卒論/Koga_code/data/d2v/K562,el=0,er=0,pl=0,pr=10000,kmer=random,N=5.d2v")
	d2v_1_model = Doc2Vec.load(d2v_1)
	d2v_5_model = Doc2Vec.load(d2v_5)

	print(d2v_1_model.dv["enhancer_0"][:10])
	print(d2v_5_model.dv["enhancer_0"][:10])

	print(d2v_1_model.wv["aaa"][:10])
	print(d2v_5_model.wv["aaa"][:10])

	print(d2v_1_model.wv.index_to_key)

test()