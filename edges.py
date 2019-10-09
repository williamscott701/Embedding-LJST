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

from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

result = elmo.embed_sentence(words) # taking only second layer

words_embeddings = {}
for idx, i in enumerate(result[2]):
    words_embeddings[words[idx]] = i

### Embeddings

embeddings_index = None

words_with_embeddings = words_embeddings.keys()

edge_embeds_multi = []
for i, j in combinations(words_with_embeddings, 2):
    edge_embeds_multi.append((words_embeddings[i], words_embeddings[j]))

n_cores = 20

pool = multiprocessing.Pool(n_cores)
embeddings_cosines = pool.map(my_utils.get_cosine_multi, edge_embeds_multi)
pool.close()

edge_embeddings = {}
for idx, (i, j) in enumerate(combinations(words_with_embeddings, 2)):
    edge_embeddings[(i, j)] = embeddings_cosines[idx]
    edge_embeddings[(j, i)] = embeddings_cosines[idx]

edges_threshold_all = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def get_edges_per_doc(doc):
    edges = [[] for i in range(len(edges_threshold_all))]
    edges_sim = [[] for i in range(len(edges_threshold_all))]
    for i in doc:
        for j in doc:
            if i != j and i in words_with_embeddings and j in words_with_embeddings:
                sim = edge_embeddings[(i, j)]
                for idx, edges_threshold in enumerate(edges_threshold_all):
                    if sim > edges_threshold and (vocabulary[i], vocabulary[j]) not in edges[idx] and (vocabulary[j], vocabulary[i]) not in edges[idx]:
                        edges[idx] += [(vocabulary[i], vocabulary[j])]
                        edges_sim[idx] += [(i, j, sim)]
    return edges, edges_sim

pool = multiprocessing.Pool(n_cores)
docs_edges_multi = pool.map(get_edges_per_doc, dataset['cleaned'].values)
pool.close()

pickle_out = open("docs_edges_multi.pickle","wb")
pickle.dump(docs_edges_multi, pickle_out, protocol=2)
pickle_out.close()