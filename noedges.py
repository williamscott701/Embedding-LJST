### Imports

from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_embedding import BertEmbedding
from allennlp.commands.elmo import ElmoEmbedder

from transformers import *
import torch
import keras

import imp, gzip
import pickle, nltk
import gensim
import multiprocessing
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as my_utils

### Definitions

def get_edges(i):
    t = np.where(i>0)[0]
    comb = combinations(t, 2)
    embeds = {j:[] for j in t}

    for p, q in comb:
        if word_similarity[p][q]:
            embeds[p] += [q]
            embeds[q] += [p]
    return embeds

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def process_df(df):
    df['text'] = my_utils.preprocess(df['text'])
    return df

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def get_edges_transformers(text):
    sentence = text.split(" ")

    if embedding_name == 'bert':
        results = bert_embedding(sentence)
        embed_vecs = np.array([i[1][0] for i in results])
    else:
        embed_vecs = elmo.embed_sentence(sentence)[2]

    l = np.array(list(set(sentence).intersection(words)))

    pp = np.array([i[1] for i in nltk.pos_tag(l)])
    pp[pp=='JJ'] = 1
    pp[pp=='JJR'] = 1
    pp[pp=='JJS'] = 1
    pp[pp=='NN'] = 1
    pp[pp=='NNS'] = 1
    pp[pp=='NNP'] = 1
    pp[pp=='NNPS'] = 1
    pp[pp!='1'] = 0
    pp = pp.astype(int)

    l = l[pp==1]

    word_embeddings = np.array([embed_vecs[sentence.index(i)] for i in l])

    word_similarity = cosine_similarity(word_embeddings)

    remove = np.where(word_similarity == 1)

    for i, j in zip(remove[0], remove[1]):
        word_similarity[i][j] = 0
        word_similarity[j][i] = 0

    word_similarity = word_similarity > cutoff
    word_similarity = word_similarity.astype(int)
    np.fill_diagonal(word_similarity, 0)

    inds = np.where(word_similarity==1)

    embeds = {words.index(j):[] for j in l}

    for i, j in zip(inds[0], inds[1]):
        embeds[words.index(l[i])] += [words.index(l[j])]

    return embeds

def get_tokenized_text(text):
    return [j for j in text if j in words]

### Config

dataset_names = ["imdb_reviews_20000", "amazon_movies_20000"]

min_df = 5
max_df = .5
max_features = 50000

n_cores = 35

for dataset_name in dataset_names:
    
    dataset = pd.read_pickle("datasets/"+ dataset_name + "_dataset")
    print(dataset_name, " read")
    
    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,
                                 stop_words="english", max_features=max_features,
                                 max_df=max_df, min_df=min_df)

    wordOccurenceMatrix = vectorizer.fit_transform(dataset.text.tolist()).toarray()

    words = vectorizer.get_feature_names()
    bertvocab = words + ['[CLS]', '[UNK]']
    pd.DataFrame(bertvocab).to_csv("resources/bertvocab_" + dataset_name + ".txt", header=None, index=None)

    ## Bert Embedding & Attention
    embedding_name = 'bert'

    cutoff = 0.95

    pretrained_weights = 'bert-base-uncased'

    model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)

    tokenizer = BertTokenizer(vocab_file="resources/bertvocab_" + dataset_name + ".txt", never_split=True, do_basic_tokenize=False)

    tokenized_text = [tokenizer.tokenize(i) for i in dataset.text]

#     temp = []
#     for i in tokenized_text:
#         t = [j for j in i if j in words]
#         temp.append(t)
        
    pool = multiprocessing.Pool(n_cores)
    temp = pool.map(get_tokenized_text, tokenized_text)
    pool.close()

    tokenized_text = temp

    indexed_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]

    input_ids = keras.preprocessing.sequence.pad_sequences(indexed_tokens, padding='post', dtype='long', maxlen=max([len(i) for i in indexed_tokens]))

    input_ids = torch.tensor(input_ids)

    input_ids = torch.split(input_ids, 500, dim=0)

    pad_length = [len(i) for i in indexed_tokens]

    print(dataset_name, "Start Bert...")
    idx = 0
    similar_words_bert = []
    similar_words_bert_attention = []

    for batch in tqdm(input_ids):

        all_embeddings, _, _, all_attentions = model(batch)
        idx_copy = deepcopy(idx)

        for one_embedding in all_embeddings.detach().numpy():
            word_embeddings = one_embedding[:pad_length[idx]]
            word_similarity = cosine_similarity(word_embeddings)
            remove = np.where(word_similarity == 1.000) # to remove self words coupling

            for i, j in zip(remove[0], remove[1]):
                word_similarity[i][j] = 0
                word_similarity[j][i] = 0

            word_similarity = word_similarity > cutoff
            word_similarity = word_similarity.astype(int)
            np.fill_diagonal(word_similarity, 0)

            inds = np.where(word_similarity==1)
            embeds = {words.index(j):[] for j in tokenized_text[idx]}

            for i, j in zip(inds[0], inds[1]):
                embeds[words.index(tokenized_text[idx][i])] += [words.index(tokenized_text[idx][j])]
            similar_words_bert.append(embeds)

        idx = deepcopy(idx_copy)

        for one_attentions in all_attentions[0].detach().numpy():

            one_side_edges = np.argmax(one_attentions[9], axis=1) #taking 9 layer of attention
            embeds = {words.index(j):[] for j in tokenized_text[idx]}

            for j, i in enumerate(one_side_edges[:pad_length[idx]]):
                if i < pad_length[idx]:
                    embeds[words.index(tokenized_text[idx][i])] += [words.index(tokenized_text[idx][j])]
            similar_words_bert_attention.append(embeds)
            idx += 1

    print(dataset_name, "Done")
    pickle_out = open("resources/"+ dataset_name + "_" + 'bert' + "_" + str(cutoff) + ".pickle","wb")
    pickle.dump(similar_words_bert, pickle_out)
    pickle_out.close()

    pickle_out = open("resources/"+ dataset_name + "_" + 'bert_attention'+ ".pickle","wb")
    pickle.dump(similar_words_bert_attention, pickle_out)
    pickle_out.close()