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

import utils as my_utils
from sentiment import SentimentAnalysis

### Read Data

dataset_name = "amazon_musical_topics_10_lambda_10_maxiter_20"

dataset = pd.read_pickle("datasets/datadf_amazon_musical")

count_matrix, _, vocabulary, words = my_utils.processReviews(dataset['text'].values)

ratings = dataset['sentiment'].values

### Making Edge_Dict

maxiter = 20
lambda_param = 10.0
n_topics = 10
n_sentiment = 5

folder_name = str(datetime.datetime.now()) + "_" + dataset_name
os.mkdir("dumps/"+folder_name)

edge_list = ["amazon_musical_bert_0.2",
             "amazon_musical_bert_0.3",
             "amazon_musical_bert_0.4",
             "amazon_musical_bert_0.5",
             "amazon_musical_fasttext_0.2",
             "amazon_musical_fasttext_0.3",
             "amazon_musical_fasttext_0.4",
             "amazon_musical_fasttext_0.5",
             "amazon_musical_glove_0.20",
             "amazon_musical_glove_0.30",
             "amazon_musical_glove_0.40",
             "amazon_musical_glove_0.50"]

import multiprocessing

def multiprocessing_func(edge_loc):
    docs_edges = pickle.load(open("resources/"+edge_loc+".pickle","rb"))
    sampler = lda.LdaSampler(count_matrix, ratings, docs_edges, words, vocabulary,
                         n_sentiment = n_sentiment, n_topics=n_topics, lambda_param=lambda_param)

    print "Running Sampler...", edge_loc
    sampler.run(maxiter=maxiter)
    joblib.dump(sampler, "dumps/" + folder_name + "/sampler_" + edge_loc)

pool = multiprocessing.Pool()

pool.map(multiprocessing_func, edge_list)