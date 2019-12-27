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

### Config

dataset_names = ["amazon_electronics", "amazon_home", "amazon_kindle", "amazon_movies"]
get_df_names = ['reviews_Electronics_5.json.gz', 'reviews_Home_and_Kitchen_5.json.gz', 'reviews_Kindle_Store_5.json.gz', 'reviews_Movies_and_TV_5.json.gz']

min_df = 5
max_df = .5
max_features = 50000
cutoffs = [0.3, 0.6]

n_cores = 40
n_docs = 100000

### Start
glove_embedding_dim = 300
glove_embeddings_index = loadGloveModel("nongit_resources/glove.6B.300d.txt")
fasttext_embedding_dim = 300
fasttext_embeddings_index = gensim.models.KeyedVectors.load_word2vec_format("nongit_resources/wiki-news-300d-1M.vec")

for dataset_name, get_df_name in zip(dataset_names, get_df_names):
    
    print("\n\n*************", dataset_name, get_df_name)

    dataset_ = getDF('datasets/' + get_df_name)
    dataset = dataset_.sample(n_docs*3)
    dataset = dataset.drop(columns=['reviewerID', 'asin', 'reviewerName', 'helpful', 'summary', 'unixReviewTime', 'reviewTime'])
    dataset = dataset.rename(columns={'reviewText': 'text', 'overall': 'sentiment'})

    n = int(dataset.shape[0]/n_cores)
    list_df = [dataset[i:i+n] for i in range(0, dataset.shape[0],n)]

    pool = multiprocessing.Pool(n_cores)
    processed_list_df = pool.map(process_df, list_df)
    pool.close()

    dataset = pd.concat(processed_list_df)
    dataset = dataset[dataset.text.apply(lambda x: len(x.split(" "))>5 and len(x.split(" "))<200)].sample(n_docs).reset_index().drop(columns='index')
    dataset.to_pickle("resources/"+ dataset_name + "_" + str(n_docs) + "_dataset")
    print("Dataset dumped")
    
    dataset_ = None

    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,
                                 stop_words="english", max_features=max_features,
                                 max_df=max_df, min_df=min_df)

    wordOccurenceMatrix = vectorizer.fit_transform(dataset.text.tolist()).toarray()

    words = vectorizer.get_feature_names()
    bertvocab = words + ['[CLS]', '[UNK]']
    pd.DataFrame(bertvocab).to_csv("bertvocab.txt", header=None, index=None)

    # Embeddings
    print("Glove")

    glove_word_embeddings = []

    for word in tqdm(words):
        emb = glove_embeddings_index.get(word, np.array([0]*glove_embedding_dim))
        glove_word_embeddings.append(emb.tolist())

    glove_word_embeddings = np.array(glove_word_embeddings)

    g = ['glove', glove_word_embeddings]

    print("Fasttext")
    fasttext_word_embeddings = []

    for word in tqdm(words):
        emb = np.array([0]*glove_embedding_dim)
        try:
            emb = fasttext_embeddings_index[word]
        except:
            pass
        fasttext_word_embeddings.append(emb.tolist())

    fasttext_word_embeddings = np.array(fasttext_word_embeddings)

    f = ['fasttext', fasttext_word_embeddings]

    #### Grid
    print("Grid")
    for embedding_name, word_embeddings in [g, f]:
        for cutoff in cutoffs:
            print(embedding_name, cutoff)
            word_similarity = cosine_similarity(word_embeddings)

            remove = np.where(word_similarity == 1)

            for i, j in zip(remove[0], remove[1]):
                word_similarity[i][j] = 0
                word_similarity[j][i] = 0

            word_similarity = word_similarity > cutoff
            word_similarity = word_similarity.astype(int)
            np.fill_diagonal(word_similarity, 0)

            wordOccuranceMatrixBinary = wordOccurenceMatrix.copy()
            wordOccuranceMatrixBinary[wordOccuranceMatrixBinary > 1] = 1

            pool = multiprocessing.Pool(n_cores)
            similar_words = pool.map(get_edges, wordOccuranceMatrixBinary)
            pool.close()
            pickle_out = open("resources/"+ dataset_name + "_" + str(n_docs) +"_" + embedding_name + "_" + str(cutoff) + ".pickle","wb")
            pickle.dump(similar_words, pickle_out)
            pickle_out.close()

    ## Bert Embedding & Attention
    print("Bert")
    embedding_name = 'bert'

    cutoff = 0.95

    pretrained_weights = 'bert-base-uncased'

    model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)

    tokenizer = BertTokenizer(vocab_file='bertvocab.txt', never_split=True, do_basic_tokenize=False)

    tokenized_text = [tokenizer.tokenize(i) for i in dataset.text]

    temp = []
    for i in tokenized_text:
        t = [j for j in i if j in words]
        temp.append(t)

    tokenized_text = temp

    indexed_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]

    input_ids = keras.preprocessing.sequence.pad_sequences(indexed_tokens, padding='post', dtype='long', maxlen=max([len(i) for i in indexed_tokens]))

    input_ids = torch.tensor(input_ids)

    input_ids = torch.split(input_ids, 1000, dim=0)

    pad_length = [len(i) for i in indexed_tokens]

    print("Start Bert...")
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

    print("Done")
    pickle_out = open("resources/"+ dataset_name + "_" + str(n_docs) +"_" + 'bert' + "_" + str(cutoff) + ".pickle","wb")
    pickle.dump(similar_words_bert, pickle_out)
    pickle_out.close()

    pickle_out = open("resources/"+ dataset_name + "_" + str(n_docs) +"_" + 'bert_attention'+ ".pickle","wb")
    pickle.dump(similar_words_bert_attention, pickle_out)
    pickle_out.close()