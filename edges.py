#### Imports

from collections import Counter

import copy
import nltk
import pickle
import gensim
import multiprocessing
from itertools import combinations

import numpy as np
import pandas as pd

import utils as my_utils

### Required Methods

dataset = pd.read_pickle("datasets/datadf_amazon_musical")

count_matrix, _, vocabulary, words = my_utils.processReviews(dataset['text'].values)

print count_matrix.shape

print "loading embeddings"

embeddings_index = gensim.models.KeyedVectors.load_word2vec_format("nongit_resources/wiki-news-300d-1M.vec")

words_embeddings = {}
for i in words:
    try:
        words_embeddings[i] = embeddings_index[i]
    except:
        pass
    
print len(words_embeddings)

embeddings_index = None

words_with_embeddings = words_embeddings.keys()

edge_embeds_multi = []
for i, j in combinations(words_with_embeddings, 2):
    edge_embeds_multi.append((words_embeddings[i], words_embeddings[j]))

len(edge_embeds_multi)

n_cores = 30

pool = multiprocessing.Pool(n_cores)
embeddings_cosines = pool.map(my_utils.get_cosine_multi, edge_embeds_multi)
pool.close()

print "loading edge_embeddings"

edge_embeddings = {}
for idx, (i, j) in enumerate(combinations(words_with_embeddings, 2)):
    edge_embeddings[(i, j)] = embeddings_cosines[idx]
    edge_embeddings[(j, i)] = embeddings_cosines[idx]

def get_edges_per_doc(doc):
    edges, edges_all = [], []
    for i in doc:
        for j in doc:
            if i != j and i in words_with_embeddings and j in words_with_embeddings:
                sim = edge_embeddings[(i, j)]
                if sim > edges_threshold and (vocabulary[i], vocabulary[j]) not in edges and (vocabulary[j], vocabulary[i]) not in edges:
                    edges.append((vocabulary[i], vocabulary[j]))
                    edges_all.append((i, j, sim))
    return (edges, edges_all)

edges_threshold = 0.00

for edges_threshold in [0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]:
    
    print edges_threshold
    
    pool = multiprocessing.Pool(n_cores)
    docs_edges_multi = pool.map(get_edges_per_doc, dataset['cleaned'].values)
    pool.close()

    docs_edges = [i[0] for i in docs_edges_multi]

    docs_edges_all = [i[1] for i in docs_edges_multi]

    pickle_out = open("resources/amazon_musical_fasttext_" + str(edges_threshold) + ".pickle","wb")
    pickle.dump(docs_edges, pickle_out)
    pickle_out.close()