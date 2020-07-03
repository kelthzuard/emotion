from numpy.random import choice
from numpy import *
import numpy
import sys

import codecs
from gensim import corpora
#from CHDP import HDP_gibbs_sampling
from CHDP import HDP_gibbs_sampling
if __name__ == "__main__":
    # corpus = codecs.open("test.txt", 'r', encoding='utf8').read().splitlines() # toy data set to test the algorithm (1001 documents)
    # voca = vocabulary.Vocabulary(excluds_stopwords=False) # find the unique words in the dataset
    # docs = [voca.doc_to_ids(doc) for doc in corpus] # change words of the corpus to ids

    train = []
    fp = codecs.open('toy_dataset.txt', 'r', encoding='utf-8')
    for line in fp:
        line = line.split()
        train.append([w for w in line])
    iterations = 3  # number of iterations for getting converged
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2idx(text) for text in train]
    HDP = HDP_gibbs_sampling(K0=20, alpha=0.01, beta=0.01, gamma=0.05, docs=corpus, V=len(dictionary))  # initialize the HDP
    for i in range(iterations):
        HDP.inference(i)

    (d, len) = HDP.worddist()  # find word distribution of each topic
    print("主题数数量：{}".format(len))
    for i in range(len):
        ind = numpy.argpartition(d[i], -10)[-10:]  # top 10 most occured words for each topic
        for j in ind:
            print(dictionary[j], ' ', end=""),
        print()

