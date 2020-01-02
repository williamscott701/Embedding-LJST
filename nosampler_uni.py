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


grid = [['amazon_electronics_100000_dataset', 'amazon_electronics_100000_bert_0.95', 1.0],
        ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_bert_attention', 1.0],
        ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_fasttext_0.3', 1.0],
        ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_fasttext_0.6', 1.0],
        ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_glove_0.3', 1.0],
        ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_glove_0.6', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_bert_0.95', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_bert_attention', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_fasttext_0.3', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_fasttext_0.6', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_glove_0.3', 1.0],
        ['amazon_home_100000_dataset', 'amazon_home_100000_glove_0.6', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_bert_0.95', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_bert_attention', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_fasttext_0.3', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_fasttext_0.6', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_glove_0.3', 1.0],
        ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_glove_0.6', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_bert_0.95', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_bert_attention', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_fasttext_0.3', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_fasttext_0.6', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_glove_0.3', 1.0],
        ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_glove_0.6', 1.0]]

        
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_bert_0.95', 0.0],
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_bert_attention', 0.0],
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_fasttext_0.3', 0.0],
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_fasttext_0.6', 0.0],
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_glove_0.3', 0.0],
#         ['amazon_electronics_100000_dataset', 'amazon_electronics_100000_glove_0.6', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_bert_0.95', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_bert_attention', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_fasttext_0.3', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_fasttext_0.6', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_glove_0.3', 0.0],
#         ['amazon_home_100000_dataset', 'amazon_home_100000_glove_0.6', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_bert_0.95', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_bert_attention', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_fasttext_0.3', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_fasttext_0.6', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_glove_0.3', 0.0],
#         ['amazon_kindle_100000_dataset', 'amazon_kindle_100000_glove_0.6', 0.0],
#         ['amazon_movies_100000_dataset', 'amazon_movies_100000_fasttext_0.3', 0.0],
#         ['amazon_movies_100000_dataset', 'amazon_movies_100000_fasttext_0.6', 0.0],
#         ['amazon_movies_100000_dataset', 'amazon_movies_100000_glove_0.3', 0.0],
#         ['amazon_movies_100000_dataset', 'amazon_movies_100000_glove_0.6', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_bert_0.95', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_bert_attention', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_fasttext_0.3', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_fasttext_0.6', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_glove_0.3', 0.0],
#         ['imdb_reviews_47405_dataset', 'imdb_reviews_47405_glove_0.6', 0.0],]
def get_scores(name, sampler):
    
    ss = silhouette_score(euclidean_distances(sampler.wordOccuranceMatrix),
                 sampler.dt_distribution.argmax(axis=1), metric='precomputed')
    
    dbs = davies_bouldin_score(sampler.wordOccuranceMatrix, sampler.dt_distribution.argmax(axis=1))
    
    chs = my_utils.coherence_score(sampler.wordOccuranceMatrix, list(sampler.getTopKWords(5).values()), sampler.vocabulary)
    
    hsc = my_utils.get_hscore_multi(sampler.dt_distribution, sampler.wordOccuranceMatrix, n_topics, 2000)
    
    loli = sampler.loglikelihood()
    
    pxy = sampler.perplexity()
        
    print("##", name, ss, dbs, chs, hsc, loli, pxy)
    
def process_sampler(inp):
    
    dataset_name = inp[0]
    embedding_name = inp[1]
    lambda_param = inp[2]
    
    print(embedding_name, "entered")

    dataset = pd.read_pickle("resources/"+ dataset_name)
    similar_words = pickle.load(open("resources/"+ embedding_name + ".pickle","rb"))
    
    min_df = 5
    max_df = .5
    maxIters = 5

    beta = .01
    gamma = 10
    n_topics = 5
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
    
    print(embedding_name, "started initialising")
    sampler._initialize_(reviews = dataset.text.tolist(), labels = dataset.sentiment.tolist())

    print(embedding_name, "started sampler")
    sampler.run(name=embedding_name, reviews=dataset.text.tolist(), labels=dataset.sentiment.tolist(), 
                similar_words=similar_words, mrf=True, maxIters=maxIters)
    
    get_scores(embedding_name, sampler)

    joblib.dump(sampler, "dumps/Uni_sampler_" + embedding_name + "_" + str(lambda_param))
    print(embedding_name, "dumped")    

pool = multiprocessing.Pool(45)
pool.map(process_sampler, grid)
pool.close()