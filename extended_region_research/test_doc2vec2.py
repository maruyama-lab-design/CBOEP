from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np

import utils

tag_and_seq_list = [
    [1, "acgtacgtacgtacgtacgtacgtacgtacgt"],
    [2, "cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgc"],
    [3, "atatatatatatttatattaattattatatatat"]
]

class SentencesIterator():
    def __init__(self, tag_and_seq_list, N):
        self.tag_and_seq_list = tag_and_seq_list
        self.N = N
        self.i = 0

    def __iter__(self):
        for tag_and_seq in self.tag_and_seq_list:
            # self.i += 1
            # print(self.i)
            for _ in range(self.N):
                yield TaggedDocument(utils.make_random_kmer_list(3, 6, tag_and_seq[1]), [tag_and_seq[0]])

sentences_1 = SentencesIterator(tag_and_seq_list, N=1)
sentences_5 = SentencesIterator(tag_and_seq_list, N=5)

model_1 = Doc2Vec(sentences_1,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
model_5 = Doc2Vec(sentences_5,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)

print(model_1.dv[1])
print(model_5.dv[1])


sentences_1_list = list(sentences_1)
sentences_5_list = list(sentences_5)

model_1_list = Doc2Vec(sentences_1_list,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
model_5_list = Doc2Vec(sentences_5_list,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)

print(model_1_list.dv[1])
print(model_5_list.dv[1])

print("__________________________________________________________")

tag_and_seq_list_1 = [
    [1, "aa", "cc", "gg", "tt", "aa", "cc", "gg", "tt"],
    [1, "a", "c", "g", "t", "a", "c", "g", "t"],
    [1, "aa", "cc", "tt", "tt", "aa", "cc", "tt", "tt"],
    [1, "a", "c", "t", "t", "a", "c", "t", "t"],
    [2, "This", "is", "a", "noise", "text","."]
]

tag_and_seq_list_2 = [
    [1, "aa", "cc", "gg", "tt", "aa", "cc", "gg", "tt"],
    # [1, "a", "c", "g", "t", "a", "c", "g", "t"],
    # [1, "aa", "cc", "gg", "tt", "aa", "cc", "gg", "tt"],
    # [1, "a", "c", "g", "t", "a", "c", "g", "t"],
    [2, "This", "is", "a", "noise", "text","."]
]

non_list = [
    [1, ""],
    [2, ""]
]

class SentencesIterator2():
    def __init__(self, tag_and_seq_list):
        self.tag_and_seq_list = iter(tag_and_seq_list)

    def __iter__(self):
        return self

    def __next__(self):
        sentence = next(self.tag_and_seq_list)
        if sentence == None:
            StopIteration()
        else:
            # print(sentence[0])
            print(sentence[1:])
            return TaggedDocument(sentence[1:], [sentence[0]])


sentences_1 = SentencesIterator2(tag_and_seq_list_1)
sentences_2 = SentencesIterator2(tag_and_seq_list_2)
sentences_non = SentencesIterator2(non_list)

model_1 = Doc2Vec(sentences_1,min_count=1,window=2,vector_size=10, sample=1e-4, epochs=5)
model_2 = Doc2Vec(sentences_2,min_count=1,window=2,vector_size=10, sample=1e-4, epochs=5)
model_non = Doc2Vec(sentences_non,min_count=1,window=2,vector_size=10, sample=1e-4, epochs=5)

print(model_1.dv[1])
print(model_2.dv[1])
print(model_non.dv[1])

class SentencesIterator3():
    def __init__(self, tag_and_seq_list):
        self.tag_and_seq_list = tag_and_seq_list

    def __iter__(self):
        for tag_and_seq in self.tag_and_seq_list:
            # print(tag_and_seq[0])
            yield TaggedDocument(tag_and_seq[1:], [tag_and_seq[0]])

sentences_1 = SentencesIterator3(tag_and_seq_list_1)
sentences_2 = SentencesIterator3(tag_and_seq_list_2)

model_1 = Doc2Vec(sentences_1,min_count=1,window=2,vector_size=10, sample=1e-4, epochs=5)
model_2 = Doc2Vec(sentences_2,min_count=1,window=2,vector_size=10, sample=1e-4, epochs=5)

print(model_1.dv[1])
print(model_2.dv[1])



