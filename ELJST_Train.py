#### Imports

from scipy import spatial, sparse
from scipy.stats import chi2
from collections import Counter
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.externals import joblib 
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import imp
import gzip
import copy
import nltk
import pickle
import scipy
import string
import gensim
import operator
import datetime

import numpy as np
import pandas as pd
import LDA_ELJST as lda
import matplotlib.pyplot as plt

import imp
import multiprocessing

import utils as my_utils
from sentiment import SentimentAnalysis

### Read Data

dataset_name = "amazon_musical"

dataset = pd.read_pickle("datasets/datadf_amazon_musical")

dataset.head(2)

count_matrix, _, vocabulary, words = my_utils.processReviews(dataset['text'].values)

Counter(dataset['sentiment'])

barren = np.where(count_matrix.sum(1)==0)[0]

ratings = dataset['sentiment'].values

### Making Edge_Dict

pickle_in = open("resources/edges_amazon_musical_glove_nontrained_0.55.pickle","rb")
docs_edges = pickle.load(pickle_in)

docs_edges = np.delete(docs_edges, barren).tolist()

len(docs_edges)

edge_dict__ = []
for doc in docs_edges:
    edge_dict_ = {}
    for i, j in doc:
        try:
            edge_dict_[i] += [j]
        except:
            edge_dict_[i] = [j]
        try:
            edge_dict_[j] += [i]
        except:
            edge_dict_[j] = [i]
    edge_dict__.append(edge_dict_)

len(edge_dict__[0])

## Run Model

maxiter = 20
lambda_param = 1.0
N_SENTIMENT = 5
n_topics = 5

folder_name = str(datetime.datetime.now()) + "_" + dataset_name
os.mkdir("dumps/"+folder_name)

topics_grid = [5, 10, 15, 20, 30, 40, 60, 100]

def multiprocessing_func(k):
    sampler = lda.LdaSampler(count_matrix, ratings, edge_dict__, n_sentiment = N_SENTIMENT, n_topics=k, lambda_param=lambda_param)

    print "Running Sampler...", k
    sampler.run(maxiter=maxiter)
    joblib.dump(sampler, "dumps/" + folder_name + "/sampler_" + dataset_name + "_n_topics_" + str(k))

pool = multiprocessing.Pool()

pool.map(multiprocessing_func, topics_grid)