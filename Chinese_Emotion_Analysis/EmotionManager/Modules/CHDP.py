"(C) Copyright 2016, Hesam Amoualian"
# References :
# D.M. Blei, A. Ng, M.I. Jordan. Latent Dirichlet Allocation. NIPS, 2002
# G. Heinrich. "Infinite LDA" -- implementing the HDP with minimum code complexity. TN2011/1, www.arbylon.net/publications/ilda.pdf
# Y.W. Teh, M.I. Jordan, M.J. Beal, D.M. Blei. Hierarchical Dirichlet Processes. JASA, 101:1566-1581, 2006

# use python3 for running and write Python3 HDPcode.py
# needs to have toy_dataset.text and vocabulary.py in same path


from gensim import corpora
import numpy, codecs
from datetime import datetime
# import vocabulary
import time
from numpy.random import choice
from numpy import *
import functools
import sys

sys.setrecursionlimit(10000)
import math


class HDP_gibbs_sampling:
    def __init__(self, K0=20, alpha=0.01, beta=0.01, gamma=0.01, docs=None, V=None):
        self.maxnn = 1
        self.alss = []  # an array for keep the stirling(N,1:N) for saving time consumming
        self.K = K0  # initial number of topics
        self.alpha = alpha  # parameter of topics prior
        self.beta = beta  # parameter of words prior
        self.gamma = gamma  # parameter of tables prior
        self.docs = docs  # a list of documents which include the words
        self.V = V  # number of different words in the vocabulary
        self.z_m_n = {}  # topic assignements for documents
        self.n_m_z = numpy.zeros((len(self.docs), self.K))  # number of words assigned to topic z in document m
        self.theta = numpy.zeros((len(self.docs), self.K))
        self.n_z_t = numpy.zeros((self.K, V))  # number of times a word v is assigned to a topic z
        self.phi = numpy.zeros((self.K, V))
        self.n_w_d = numpy.zeros((self.K, V))   #add
        self.n_w = numpy.zeros((self.K))        #add
        self.n_z = numpy.zeros(self.K)  # total number of words assigned to a topic z
        self.U1 = []  # active topics
        for i in range(self.K):
            self.U1.append(i)

        self.U0 = []  # deactive topics
        self.tau = numpy.zeros(self.K + 1) + 1. / self.K
        for m, doc in enumerate(docs):  # Initialization of the data structures
            for n, t in enumerate(doc):
                z = numpy.random.randint(0, self.K)  # Randomly assign a topic to a word and increase the counting array
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
                self.n_w_d[z, t] += 1    #add
                self.n_w[z] += 1           #add
                self.z_m_n[(m, n)] = z

    def perplexity(self, iteration):  # 计算混淆度
        per = 0;
        for m, doc in enumerate(self.docs):
            for n, t in enumerate(doc):
                totpro = 0;
                for kk in range(self.K):  # using the z sampling equation in ILDA
                    k = self.U1[kk]
                    totpro += (self.n_m_z[m, k] + self.alpha * self.tau[k]) * (self.n_z_t[k, t] + self.beta) / (
                            self.n_z[k] + self.V * self.beta) * (self.K * self.alpha * self.tau[k] + self.z_m_n[(m, n)])
                per -= numpy.log(totpro)

        per = numpy.exp(per / self.V)
        print('Iteration:', iteration, numpy.around(per, decimals=3));

    def inference(self, iteration):
        " Inference of HDP  using Dircet Assignment with ILDA simpilifying "
        p=2    #基于中心词向前扩展p个词
        q=2    #基于中心词向后扩展p个词
        for m, doc in enumerate(self.docs):
            for n, t in enumerate(doc):
                # decrease the counting for word t with topic kold
                kold = self.z_m_n[(m, n)]
                self.n_m_z[m, kold] -= 1
                self.n_z[kold] -= 1
                p_z = numpy.zeros(self.K + 1)
                self.n_w[kold] -= 1  # add
                self.n_z_t[kold, t] -= 1
                self.n_w_d[kold, t] -= 1  # add
                for i in range(t-p,t+q) :
                    for kk in range(self.K):  # using the z sampling eqation in ILDA
                        k = self.U1[kk]
                        p_z[kk] = (self.n_m_z[m, k]  + self.alpha * self.tau[k]) * (2*self.n_z_t[k, t] + self.n_w_d[k,t]  + self.beta) / (
                               self.n_z[k] + self.n_w[k]  + self.V * self.beta)
                p_z[self.K] = (self.alpha * self.tau[self.K]) / self.V  # additional cordinate for new topic
                knew = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                if knew == self.K:  # check if topic sample is new
                    self.z_m_n[(m, n)] = self.spawntopic(m,t)  # extend the number of topics and arrays shape and assign the array for new topic
                    self.updatetau()  # update the table distribution over topic

                else:
                    k = self.U1[knew]  # do same as LDA
                    self.z_m_n[(m, n)] = k
                    self.n_m_z[m, k] += 1
                    self.n_z_t[k, t] += 1
                    self.n_z[k] += 1
                    self.n_w_d[k, t] += 1  # add
                    self.n_w[k] += 1  # add

                if self.n_z[kold] == 0:  # check if the topic have been not used and re shape the arrayes
                    self.U1.remove(kold)
                    self.U0.append(kold)
                    self.K -= 1
                    self.updatetau()

    # print ('Iteration:',iteration,'\n','Number of topics:',self.K,'\n','Activated topics:',self.U1,'\n','Deactivated topics',self.U0)

    def spawntopic(self, m, t):  # reshape the arrays for new topic
        if len(self.U0) > 0:  # if the we have deactive topics.
            k = self.U0[0]
            self.U0.remove(k)
            self.U1.append(k)
            self.n_m_z[m, k] = 1
            self.n_z_t[k, t] = 1
            self.n_z[k] = 1
            self.n_w_d[k, t] = 1  # add
            self.n_w[k] = 1  # add


        else:
            k = self.K  # if the we do not have deactive topics so far.
            self.n_m_z = numpy.append(self.n_m_z, numpy.zeros([len(self.docs), 1]), 1)
            self.U1.append(k)
            self.n_m_z[m, k] = 1
            self.n_z_t = numpy.vstack([self.n_z_t, numpy.zeros(self.V)])
            self.n_z_t[k, t] = 1
            self.n_w_d = numpy.vstack([self.n_w_d, numpy.zeros(self.V)])     # add
            self.n_w_d[k, t] = 1  # add
            self.n_w = numpy.append(self.n_w, 1)  # add
            self.n_z = numpy.append(self.n_z, 1)
            self.tau = numpy.append(self.tau, 0)

        self.K += 1

        return k

    def stirling(self, nn):  # making an array for keep the stirling(N,1:N) for saving time consumming
        if len(self.alss) == 0:
            self.alss.append([])
            self.alss[0].append(1)
        if nn > self.maxnn:
            for mm in range(self.maxnn, nn):
                ln = len(self.alss[mm - 1]) + 1
                self.alss.append([])

                for xx in range(ln):
                    self.alss[mm].append(0)
                    if xx < (ln - 1):
                        self.alss[mm][xx] += self.alss[mm - 1][xx] * mm
                    if xx > (ln - 2):
                        self.alss[mm][xx] += 0
                    if xx == 0:
                        self.alss[mm][xx] += 0
                    if xx != 0:
                        self.alss[mm][xx] += self.alss[mm - 1][xx - 1]

            self.maxnn = nn
        return self.alss[nn - 1]

    def rand_antoniak(self, alpha, n):
        # Sample from Antoniak Distribution
        ss = self.stirling(n)
        max_val = max(ss)
        p = numpy.array(ss) / max_val

        aa = 1
        for a, _ in enumerate(p):
            p[a] *= aa
            aa *= alpha

        p = numpy.array(p, dtype='float') / numpy.array(p, dtype='float').sum()
        return choice(range(1, n + 1), p=p)

    def updatetau(self):  # update tau using antoniak sampling from CRM

        m_k = numpy.zeros(self.K + 1)
        for kk in range(self.K):
            k = self.U1[kk]
            for m in range(len(self.docs)):

                if self.n_m_z[m, k] > 1:
                    m_k[kk] += self.rand_antoniak(self.alpha * self.tau[k], int(self.n_m_z[m, k]))
                else:
                    m_k[kk] += self.n_m_z[m, k]

        T = sum(m_k)
        m_k[self.K] = self.gamma
        tt = numpy.transpose(numpy.random.dirichlet(m_k, 1))
        for kk in range(self.K):
            k = self.U1[kk]
            self.tau[k] = tt[kk]

        self.tau[self.K] = tt[self.K]

    def worddist(self):
        """topic-word distribution, \phi in Blei'spaper  """
        #return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + self.V * self.beta), len(self.n_z)
        return (self.n_z_t + self.n_w_d + self.beta) / (self.n_z[:, numpy.newaxis] + self.n_w[:, numpy.newaxis] + self.V * self.beta), len(self.n_z)





