from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np

sentences_1 = [
    [1, "aaa", "ccc", "ggg", "ttt"],
    [1, "a", "c", "g", "t"],
    [2, "t", "g", "a", "c"],
    [2, "ttt", "ggg", "aaa", "ccc"],
]

sentences_2 = [
    [1, "a", "c", "g", "t"],
    # [1, "a", "c", "g", "t"],
    [2, "t", "g", "a", "c"],
    # [2, "t", "g", "a", "c"],
]

def read_corpus(sentences):
    for sentence in sentences:
        yield TaggedDocument(sentence[1:], [sentence[0]])


corpus_1 = list(read_corpus(sentences_1))
corpus_2 = list(read_corpus(sentences_2))
model_1 = Doc2Vec(documents=corpus_1,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
print(model_1.dv[1])
model_2 = Doc2Vec(documents=corpus_2,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
print(model_2.dv[1])

print("___________________________")
class Corpus():
    def __init__(self, sentences):
        self.sentences = iter(sentences)

    def __iter__(self):
        for sentence in self.sentences:
            print(sentence[0])
            yield TaggedDocument(sentence[1:], [sentence[0]])
        # return self

    # def __next__(self):
    #     sentence = next(self.sentences)
    #     if sentence == None:
    #         StopIteration()
    #     else:
    #         print(sentence[0])
    #         return TaggedDocument(sentence[1:], [sentence[0]])

def corpus_generator(sentences):
    for sentence in sentences:
        yield TaggedDocument(sentence[1:], [sentence[0]])

class SentencesIterator():
    # 参考 https://jacopofarina.eu/posts/gensim-generator-is-not-iterator/
    def __init__(self, generator_function, sentences):
        self.generator_function = generator_function
        self.sentences = sentences
        self.generator = self.generator_function(self.sentences)

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function(self.sentences)
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

corpus_1 = SentencesIterator(corpus_generator, sentences_1)
corpus_2 = SentencesIterator(corpus_generator, sentences_2)
model_1 = Doc2Vec(corpus_1, min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
# model_1.build_vocab(corpus_1)
# model_1.train(
#     corpus_1,
#     total_examples=model_1.corpus_count,
#     epochs=model_1.epochs
# )
print(model_1.dv[1])
model_2 = Doc2Vec(corpus_2,min_count=1,window=10,vector_size=10, sample=1e-4,negative=5, epochs=5)
print(model_2.dv[1])