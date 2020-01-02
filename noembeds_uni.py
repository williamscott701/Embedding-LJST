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


grid = ["amazon_electronics_20000", "amazon_home_20000", "amazon_kindle_20000",
        "amazon_movies_20000", "imdb_reviews_20000", "twitter_airline_9061"]

def process_sampler(dataset_name):
    
    print(dataset_name, "entered")

    dataset = pd.read_pickle("datasets/"+ dataset_name + "_dataset")
    
    min_df = 5
    max_df = .5
    maxIters = 20

    lambda_param = 1.0
    beta = .01
    gamma = 10
    n_topics = 5
    n_sentiment = dataset.sentiment.unique().shape[0]

    alpha = 0.1/n_topics * np.ones(n_topics)
    gamma = [gamma/(n_topics*n_sentiment)]*n_sentiment

    similar_words = [{} for i in range(dataset.shape[0])]

    sampler = lda.SentimentLDAGibbsSampler(n_topics, alpha, beta, gamma, numSentiments=n_sentiment, SentimentRange = n_sentiment, max_df = max_df, min_df = min_df, lambda_param = lambda_param)
    
    sampler._initialize_(reviews = dataset.text.tolist(), labels = dataset.sentiment.tolist())

    sampler.run(name=dataset_name, reviews=dataset.text.tolist(), labels=dataset.sentiment.tolist(), 
                similar_words=similar_words, mrf=True, maxIters=maxIters, debug=False)
    
    joblib.dump(sampler, "dumps/Uni_sampler_" + dataset_name + "_noembeds")
    print(dataset_name, "dumped")
    
pool = multiprocessing.Pool(10)
pool.map(process_sampler, grid)
pool.close()

# process_sampler(grid[0])