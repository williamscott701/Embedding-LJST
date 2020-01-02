from tqdm import tqdm
from collections import Counter
from itertools import combinations
from nltk.stem import PorterStemmer
import joblib 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances

import imp
import copy
import pickle
import multiprocessing

import numpy as np
import pandas as pd
import utils as my_utils
import ELJST_script_unigram as lda
import matplotlib.pyplot as plt



grid = [['amazon_electronics_20000', 'bert_0.95'],
        ['amazon_electronics_20000', 'bert_attention'],
        ['amazon_electronics_20000', 'fasttext_0.3'],
        ['amazon_electronics_20000', 'fasttext_0.6'],
        ['amazon_electronics_20000', 'glove_0.3'],
        ['amazon_electronics_20000', 'glove_0.6'],
        
        ['amazon_home_20000', 'bert_0.95'],
        ['amazon_home_20000', 'bert_attention'],
        ['amazon_home_20000', 'fasttext_0.3'],
        ['amazon_home_20000', 'fasttext_0.6'],
        ['amazon_home_20000', 'glove_0.3'],
        ['amazon_home_20000', 'glove_0.6'],
        
        ['amazon_kindle_20000', 'bert_0.95'],
        ['amazon_kindle_20000', 'bert_attention'],
        ['amazon_kindle_20000', 'fasttext_0.3'],
        ['amazon_kindle_20000', 'fasttext_0.6'],
        ['amazon_kindle_20000', 'glove_0.3'],
        ['amazon_kindle_20000', 'glove_0.6'],
        
        ['amazon_movies_20000', 'bert_0.95'],
        ['amazon_movies_20000', 'bert_attention'],
        ['amazon_movies_20000', 'fasttext_0.3'],
        ['amazon_movies_20000', 'fasttext_0.6'],
        ['amazon_movies_20000', 'glove_0.3'],
        ['amazon_movies_20000', 'glove_0.6'],
        
        ['imdb_reviews_20000', 'bert_0.95'],
        ['imdb_reviews_20000', 'bert_attention'],
        ['imdb_reviews_20000', 'fasttext_0.3'],
        ['imdb_reviews_20000', 'fasttext_0.6'],
        ['imdb_reviews_20000', 'glove_0.3'],
        ['imdb_reviews_20000', 'glove_0.6'],
        
        ['twitter_airline_9061', 'bert_0.95'],
        ['twitter_airline_9061', 'bert_attention'],
        ['twitter_airline_9061', 'fasttext_0.3'],
        ['twitter_airline_9061', 'fasttext_0.6'],
        ['twitter_airline_9061', 'glove_0.3'],
        ['twitter_airline_9061', 'glove_0.6']]


def process_sampler(inp):
    
    dataset_name = inp[0]
    embedding_name = inp[1]
    
    print(dataset_name, embedding_name, "entered")

    dataset = pd.read_pickle("datasets/" + dataset_name + "_dataset")
    similar_words = pickle.load(open("resources/" + dataset_name + "_" + embedding_name + ".pickle","rb"))
    
    min_df = 5
    max_df = .5
    maxIters = 20

    beta = .01
    gamma = 10
    n_topics = 5
    lambda_param = 1.0
    n_sentiment = dataset.sentiment.unique().shape[0]

    alpha = 0.1/n_topics * np.ones(n_topics)
    gamma = [gamma/(n_topics*n_sentiment)]*n_sentiment

    for s in similar_words:
        for i in s.keys():
            k = list(set(s[i]))
            if i in k:
                k.remove(i)
            s[i] = k

    sampler = lda.SentimentLDAGibbsSampler(n_topics, alpha, beta, gamma, numSentiments=n_sentiment, SentimentRange = n_sentiment, max_df = max_df, min_df = min_df, lambda_param = lambda_param)
    
    sampler._initialize_(reviews = dataset.text.tolist(), labels = dataset.sentiment.tolist())

    print(dataset_name, embedding_name, "started sampler")
    sampler.run(name=embedding_name, reviews=dataset.text.tolist(), labels=dataset.sentiment.tolist(), 
                similar_words=similar_words, mrf=True, maxIters=maxIters, debug=False)

    joblib.dump(sampler, "dumps/Uni_sampler_" + dataset_name + "_" + embedding_name)
    print(dataset_name, embedding_name, "dumped")   

pool = multiprocessing.Pool(40)
pool.map(process_sampler, grid)
pool.close()

# process_sampler(grid[0])