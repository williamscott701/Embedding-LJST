from scipy import spatial, sparse
from scipy.stats import chi2
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup as bs

import copy
import nltk
import scipy
import multiprocessing

import numpy as np
import pandas as pd

stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def convert_numbers(k):
    for i in range(len(k)):
        try:
            num2words(int(k[i]))
            k[i] = " "
        except:
            pass
    return k

def preprocess_with_nums(pd):
    pd = pd.str.lower()
    pd = pd.str.replace('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t'), ' ')
    pd = pd.apply(lambda x: [w for w in w_tokenizer.tokenize(x)])
#     pd = pd.apply(lambda x: convert_numbers(x))
    pd = pd.str.join(' ')
    
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])    
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w, 'v') for w in x])
    pd = pd.apply(lambda x: [item for item in x if item not in stop_words])
    pd = pd.apply(lambda x: [item for item in x if len(item)>1])
    return pd


def preprocess(pd):
    pd = pd.str.lower()
    pd = pd.str.replace('[^a-zA-Z]', ' ')
    pd = pd.apply(lambda x: [w for w in w_tokenizer.tokenize(x)])
    pd = pd.str.join(' ')
    
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])    
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w, 'v') for w in x])
    pd = pd.apply(lambda x: [item for item in x if item not in stop_words])
    pd = pd.apply(lambda x: [item for item in x if len(item)>3])
    pd = pd.apply(lambda x: [i[0] for i in nltk.pos_tag(x) if i[1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']])
    pd = pd.apply(lambda x: " ".join(x))
    return pd

# def preprocess(pd):
#     pd = pd.str.lower()
#     pd = pd.str.replace('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t'), ' ')
#     pd = pd.str.replace('\d+', ' ')
#     pd = pd.apply(lambda x: [w for w in w_tokenizer.tokenize(x)])
#     pd = pd.apply(lambda x: convert_numbers(x))
#     pd = pd.str.join(' ')
    
#     pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])    
#     pd = pd.apply(lambda x: [lemmatizer.lemmatize(w, 'v') for w in x])
#     pd = pd.apply(lambda x: [item for item in x if item not in stop_words])
#     pd = pd.apply(lambda x: [item for item in x if len(item)>2])
#     pd = pd.apply(lambda x: " ".join(x))
#     return pd

def preprocess_lite(pd):
    pd = pd.str.lower()
    pd = pd.str.replace('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t'), ' ')
    pd = pd.apply(lambda x: [w for w in w_tokenizer.tokenize(x)])
    pd = pd.apply(lambda x: convert_numbers(x))
    pd = pd.str.join(' ')
    
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])    
    pd = pd.apply(lambda x: [item for item in x if len(item)>1])
    return pd

def processReviews(reviews, window=5, MAX_VOCAB_SIZE=50000):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, max_df=0.7, min_df = 7, max_features=MAX_VOCAB_SIZE)
    count_matrix = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    vocabulary = dict(zip(words,np.arange(len(words))))
    inv_vocabulary = dict(zip(np.arange(len(words)),words))
    return count_matrix.toarray(), vocabulary, words

def get_cosine(a, b):
    return 1 - spatial.distance.cosine(a, b)

def get_cosine_multi(a):
    return 1 - spatial.distance.cosine(a[0], a[1])

def print_if_mod(idx, n):
    if idx % n == 0:
        print(idx)
        
def extract_body(m):
    b = bs(m)
    t = b.find_all('code')
    for tag in t:
        tag.replace_with('')
    a = list({tag.name for tag in b.find_all()})
    for i in a:
        b = str(b).replace("<"+str(i)+">", " ")
        b = str(b).replace("</"+str(i)+">", " ")
    return b

def coherence_score(X, topic_sentiment_df, vocabulary):
    X[X>1] = 1    
    totalcnt = len(topic_sentiment_df)
    total = 0
    for allwords in topic_sentiment_df:
        for word1 in allwords:
            for word2 in allwords:
                if word1 != word2:
                    ind1 = vocabulary[word1]
                    ind2 = vocabulary[word2]
                    total += np.log((np.matmul(X[:,ind1].T, X[:,ind2]) + 1.0)/np.sum(X[:,ind2]))
    return total/(2*totalcnt)

def kl_score(pk,qk):
    return (scipy.stats.entropy(pk,qk)*.5 + scipy.stats.entropy(qk,pk)*.5)

def get_hscore(dt_distribution, X, k):
    testlen = X.shape[0]
    all_kl_scores = np.zeros((testlen, testlen))
    for i in range(testlen-1):
        for j in range(i+1,testlen):
            score = kl_score(dt_distribution[i],dt_distribution[j])
            all_kl_scores[i,j] = score
            all_kl_scores[j,i] = score

    dt = np.zeros((X.shape[0], k))

    for i in range(X.shape[0]):
        dt[i, dt_distribution[i].argmax()]=1

    intradist = 0
    for i in range(k):
        cnt = dt[:,i].sum()
        tmp = np.outer(dt[:,i],dt[:,i])
        tmp = tmp * all_kl_scores
        intradist += tmp.sum()*1.0/(cnt*(cnt-1))
#         print(cnt, tmp.sum(), intradist)
    intradist = intradist/k

    interdist = 0
    for i in range(k):
       for j in range(k):
           if i != j:
             cnt_i = dt[:,i].sum()
             cnt_j = dt[:,j].sum()
             tmp = np.outer(dt[:,i], dt[:,j])
             tmp = tmp * all_kl_scores
             interdist += tmp.sum()*1.0/(cnt_i*cnt_j)
    interdist = interdist/(k*(k-1))
    return intradist/interdist

def kl_score_multi(comb):
    pk, qk = comb[0], comb[1]
    return (scipy.stats.entropy(pk, qk)*.5 + scipy.stats.entropy(qk,pk)*.5)

def get_hscore_multi(dt_distribution_, X_, k, testlen):
    
    index = np.random.choice(X_.shape[0], testlen, replace=False)
    
    dt_distribution = dt_distribution_[index]
    X = X_[index]

    all_kl_scores = np.zeros((testlen, testlen))

    combinations = []
    dt_combinations = []
    for i in range(testlen-1):
        for j in range(i+1,testlen):
            combinations.append([i, j])
            dt_combinations.append((dt_distribution[i], dt_distribution[j]))

    pool = multiprocessing.Pool(10)
    scores = pool.map(kl_score_multi, dt_combinations)
    pool.close()

    for idx, (i, j) in enumerate(combinations):
        all_kl_scores[i, j] = scores[idx]
        all_kl_scores[j, i] = all_kl_scores[i, j]

    dt = np.zeros((X.shape[0], k))

    for i in range(X.shape[0]):
        dt[i, dt_distribution[i].argmax()]=1

    intradist = 0
    for i in range(k):
        cnt = dt[:,i].sum()
        tmp = np.outer(dt[:,i],dt[:,i])
        tmp = tmp * all_kl_scores
        intradist += tmp.sum()*1.0/(cnt*(cnt-1))
    #         print(cnt, tmp.sum(), intradist)
    intradist = intradist/k

    interdist = 0
    for i in range(k):
       for j in range(k):
           if i != j:
             cnt_i = dt[:,i].sum()
             cnt_j = dt[:,j].sum()
             tmp = np.outer(dt[:,i], dt[:,j])
             tmp = tmp * all_kl_scores
             interdist += tmp.sum()*1.0/(cnt_i*cnt_j)
    interdist = interdist/(k*(k-1))
    return intradist/interdist