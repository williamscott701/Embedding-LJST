# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:47:36 2018

@author: asengup6
"""

from time import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error
from itertools import combinations
from toolz import compose
from sklearn.model_selection import train_test_split
import scipy
from scipy.special import gammaln, psi
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import ast
from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm
from tqdm import trange

st = PorterStemmer()
MAX_VOCAB_SIZE = 50000
def save_document_image(filename, doc, zoom=2):
    """
    Save document as an image.
    doc must be a square matrix
    """
    height, width = doc.shape
    zoom = np.ones((width*zoom, width*zoom))
    # imsave scales pixels between 0 and 255 automatically
    scipy.misc.imsave(filename, np.kron(doc, zoom))

def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution
    """
    return np.random.dirichlet(alpha)

def sampleFromCategorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()

def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx
            
class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments, SentimentRange, max_df = .5, min_df = 5, max_features = MAX_VOCAB_SIZE, lambda_param = 1):
        """
        numTopics: Number of topics in the model
        numSentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments
        self.SentimentRange = SentimentRange
        self.probabilities_ts = {}
        self.lambda_param = lambda_param
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.wordOccuranceMatrix_shape = None
        
        # review = review.decode("ascii", "ignore").encode("ascii")

    def processReviews(self, reviews):
        #self.vectorizer = SkipGramVectorizer(analyzer="word",stop_words="english",max_features=MAX_VOCAB_SIZE,max_df=.75,min_df=10, k = window,ngram_range=(1,1))
        self.vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words="english",max_features=self.max_features,max_df=self.max_df, min_df=self.min_df)
        train_data_features = self.vectorizer.fit_transform(reviews)
        self.words = self.vectorizer.get_feature_names()
        print(len(self.words), train_data_features.toarray().shape)
        self.vocabulary = dict(zip(self.words,np.arange(len(self.words))))
        self.inv_vocabulary = dict(zip(np.arange(len(self.words)),self.words))
        wordOccurenceMatrix = train_data_features.toarray()
        return wordOccurenceMatrix
        
    def create_priorsentiment(self):
        sid = SentimentIntensityAnalyzer()
        l = []
        binsize = self.SentimentRange*1.0/self.numSentiments
        for i in self.vocabulary:
            l.append(sid.polarity_scores(i).get('compound',np.nan))
        clf = MinMaxScaler(feature_range = (0, self.numSentiments))
        l = clf.fit_transform(np.array(l))
        l = [min(int(i/binsize)-1,0) for i in l]
        self.priorSentiment = dict(zip(list(self.vocabulary.keys()),l))

    def _initialize_(self, reviews, labels, unlabeled_reviews=[]):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        allreviews = reviews + unlabeled_reviews
        self.wordOccuranceMatrix = self.processReviews(allreviews)
        self.wordOccuranceMatrix_shape = self.wordOccuranceMatrix.shape[0]
        #self.create_priorsentiment()
        numDocs, vocabSize = self.wordOccuranceMatrix.shape
        
        numDocswithlabels = len(labels)
        # Pseudocounts
        self.n_dt = np.zeros((numDocs, self.numTopics))
        self.n_dts = np.zeros((numDocs, self.numTopics, self.numSentiments))
        self.n_d = np.zeros((numDocs))
        self.n_vts = np.zeros((vocabSize, self.numTopics, self.numSentiments))
        self.vts = np.zeros((vocabSize, self.numTopics, self.numSentiments))
        self.n_ts = np.zeros((self.numTopics, self.numSentiments))
        self.dt_distribution = np.zeros((numDocs, self.numTopics))
        self.dts_distribution = np.zeros((numDocs, self.numTopics, self.numSentiments))
        self.topics = {}
        self.sentiments = {}
        self.sentimentprior = {}

        self.alphaVec = self.alpha.copy()
        self.gammaVec = self.gamma.copy()
        
        self.docs_edges = []
        self.loglikelihood_history = []
        
        for d in range(numDocs):

            if d < numDocswithlabels:
                Doclabel = labels[d]
                binsize = self.SentimentRange*1.0/self.numSentiments
                DoclabelMatrix = np.eye(self.numSentiments)*.2
                DoclabelMatrix[max(int(Doclabel/binsize)-1,0),max(int(Doclabel/binsize)-1,0)] = 1.2
                gammaVec = np.matmul(DoclabelMatrix,self.gammaVec)
                self.sentimentprior[d] = gammaVec
            else:
                self.sentimentprior[d] = np.array(self.gammaVec)
            
            topicDistribution = sampleFromDirichlet(self.alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
            for i, w in enumerate(word_indices(self.wordOccuranceMatrix[d, :])):
                t = sampleFromCategorical(topicDistribution)
                s = sampleFromCategorical(sentimentDistribution[t, :])

                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1
                self.vts[w,t,s] = 1

            self.dt_distribution[d,:] = (self.n_dt[d] + self.alphaVec) / \
            (self.n_d[d] + np.sum(self.alphaVec))
            self.dts_distribution[d,:,:] = (self.n_dts[d, :, :] + self.sentimentprior[d]) / \
                (self.n_dt[d, :] + np.sum(self.sentimentprior[d]))[:,np.newaxis]
                
    def conditionalDistribution(self, d, v, similar_words, mrf = True, debug_mode=False):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        topic_assignment1 = np.ones((self.numTopics, self.numSentiments))
        
        firstFactor = (self.n_dt[d] + self.alphaVec) / \
            (self.n_d[d] + np.sum(self.alphaVec))
        
        secondFactor = np.zeros((self.numTopics,self.numSentiments))
        for k in range(self.numTopics):
            secondFactor[k,:] = (self.n_dts[d, k, :] + self.sentimentprior[d]) / \
                (self.n_dt[d, k] + np.sum(self.sentimentprior[d]))
        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= firstFactor[:, np.newaxis]
        probabilities_ts *= secondFactor * thirdFactor
        #probabilities_ts = np.exp(probabilities_ts)
        probabilities_ts /= np.sum(probabilities_ts)
        
        if mrf == True and self.lambda_param != 0:
            all_children = np.zeros(self.wordOccuranceMatrix_shape).astype(int)
            try:
                all_children[similar_words[v]] = 1
            except:
                pass
            
            # all_children = similar_words[v,:] #[d][i1,:]
            new_C = self.vts[all_children,:, :]
            topic_assignment = new_C.sum(0)
            topic_assignment /= topic_assignment.sum()
            topic_assignment1 = np.exp(self.lambda_param * topic_assignment)

        if debug_mode == True:
            print (probabilities_ts, topic_assignment1)
        
        probabilities_ts *= topic_assignment1
        probabilities_ts /= np.sum(probabilities_ts)

        if debug_mode == True:
            print (probabilities_ts)
            
        return probabilities_ts

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        worddict = {}
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-K:]
                vocab = self.vectorizer.get_feature_names()
                worddict[(t,s)] = [vocab[i] for i in topWordIndices]
        return worddict

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.n_vts.shape[0]
        n_docs = self.n_dt.shape[0]
        lik = 0

        alpha = self.n_vts+self.beta
        lik += np.sum(gammaln(alpha).sum(0) - gammaln(alpha.sum(0)))
        lik -= (vocab_size*gammaln(self.beta) - gammaln(vocab_size*self.beta)) * self.numTopics * self.numSentiments

        sentimentprior = np.array([self.sentimentprior[i] for i in self.sentimentprior.keys()])
        n_dts = self.n_dts.copy()
        for z in range(self.numTopics):
            n_dts[:, z, :] = n_dts[:, z, :] + sentimentprior
        lik += np.sum(gammaln(n_dts).sum(2) - gammaln(n_dts.sum(2)))
        lik -= np.sum(gammaln(sentimentprior).sum(1) - gammaln(sentimentprior.sum(1))) * self.numTopics

        alpha = self.n_dt+self.alpha
        lik += np.sum(gammaln(alpha).sum(1) - gammaln(alpha.sum(1)))
        lik -= n_docs * (self.numTopics*gammaln(self.alpha.sum()) - gammaln(self.numTopics * self.alpha.sum()))

        for i in range(n_docs):
            edges_count = len(self.docs_edges[i])
            if edges_count:
                t = np.array(self.docs_edges[i])
                aa = t[:, 0]
                bb = t[:, 1]
                cc = (self.n_vts[aa, :, :].argmax(1) == self.n_vts[bb, :, :].argmax(1)).sum(0)
                lik += np.sum(np.log(np.exp(self.lambda_param*cc/edges_count)))

    #     for z in range(self.numTopics):
    #         for s in range(self.numSentiments):
    #             lik += log_multi_beta(self.n_vts[:, z, s]+self.beta)
    #             lik -= log_multi_beta(self.beta, vocab_size)

    #     for m in range(n_docs):
    #         for z in range(self.numTopics):
    #             alpha = self.n_dts[m, z, :]+self.sentimentprior[m]
    #             lik += np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    #             alpha = self.sentimentprior[m]
    #             lik -= np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    #     for m in range(n_docs):
    #         lik += log_multi_beta(self.n_dt[m,:]+self.alpha)
    #         lik -= log_multi_beta(self.alpha.sum(), self.numTopics)

    #     for i in range(n_docs):
    #         for s in range(self.numSentiments):
    #             count = 0
    #             edges_count = 0
    #             for a, b in (docs_edges[i]):
    #                 edges_count += 1
    #                 aa = self.n_vts[a, :, s]
    #                 bb = self.n_vts[b, :, s]
    #                 if aa.argmax() == bb.argmax():
    #                     count += 1
    #             if edges_count > 0:
    #                 lik += np.log(np.exp(self.lambda_param*count/edges_count))
    
        return lik
    
    def perplexity(self):
        return np.exp(-self.loglikelihood()/self.wordOccuranceMatrix.sum())

    def run(self, name, reviews, labels, similar_words, unlabeled_reviews=[], mrf = True, maxIters=100, debug=False):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        #self._initialize_(reviews, labels, unlabeled_reviews)
        self.loglikelihoods = np.zeros(maxIters)
        numDocs, vocabSize = self.wordOccuranceMatrix.shape
        
        self.docs_edges = []
        for i in similar_words:
            edges = []
            for j in i.keys():
                for p in i[j]:
                    edges.append([j, p])
            self.docs_edges.append(edges)
        
        for iteration in range(maxIters):
            print("**", name, iteration)
            loglikelihood = 0
            if debug:
                r = trange(numDocs)
            else:
                r = range(numDocs)
            for idx, d in enumerate(r):
                for i, v in enumerate(word_indices(self.wordOccuranceMatrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1
                    self.vts[v,t,s] = 0

                    probabilities_ts = self.conditionalDistribution(d, v, similar_words[idx], mrf)
                    #if v in self.priorSentiment:
                    #    s = self.priorSentiment[v]
                    #    t = sampleFromCategorical(probabilities_ts[:, s])
                    #else:
                    ind = sampleFromCategorical(probabilities_ts.flatten())
                    t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.probabilities_ts[(d,v)] = probabilities_ts[t,s]
                    #loglikelihood += np.log(self.probabilities_ts[(d,v)])
                    
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
                    self.vts[v,t,s] = 1
                #print ("end time {}".format(time()))   
                
                if iteration == maxIters - 1:
                    self.dt_distribution[d,:] = (self.n_dt[d,:] + self.alphaVec) / \
                    (self.n_d[d] + np.sum(self.alphaVec))
                    self.dts_distribution[d,:,:] = (self.n_dts[d, :, :] + self.sentimentprior[d]) / \
                    (self.n_dt[d, :] + np.sum(self.sentimentprior[d]))[:,np.newaxis]
                
                    self.dt_distribution = self.dt_distribution/np.sum(self.dt_distribution, axis=1)[:,np.newaxis]
                    self.dts_distribution = self.dts_distribution/np.sum(self.dts_distribution, axis=2)[:,:,np.newaxis]
                
                #loglikelihood += np.sum(gammaln((self.n_dt[d] + self.alphaVec))) - gammaln(np.sum((self.n_dt[d] + self.alphaVec)))
                #loglikelihood -= np.sum(gammaln(self.alphaVec)) - gammaln(np.sum(self.alphaVec))
                
                #loglikelihood += np.sum(gammaln((self.n_dts[d, :, :] + self.sentimentprior[d]))) - gammaln(np.sum(self.n_dts[d, :, :] + self.sentimentprior[d]))
                #loglikelihood -= np.sum(gammaln(self.sentimentprior[d])) - gammaln(np.sum(self.sentimentprior[d]))
            

            #loglikelihood += (np.sum(gammaln((self.n_vts + self.beta))) - gammaln(np.sum((self.n_vts + self.beta))))
            #loglikelihood -= (vocabSize * gammaln(self.beta) - gammaln(vocabSize * self.beta))

            #self.loglikelihoods[iteration] = loglikelihood 
            #print ("Total loglikelihood is {}".format(loglikelihood))
            
            if (iteration+1)%5 == 0:
                # ADJUST ALPHA BY USING MINKA'S FIXED-POINT ITERATION
                numerator = 0
                denominator = 0
                for d in range(numDocs):
                    numerator += psi(self.n_dt[d] + self.alphaVec) - psi(self.alphaVec)
                    denominator += psi(np.sum(self.n_dt[d] + self.alphaVec)) - psi(np.sum(self.alphaVec))
                
                self.alphaVec *= numerator / denominator     
                self.alphaVec = np.maximum(self.alphaVec,self.alpha)
            self.loglikelihood_history.append(self.loglikelihood())
                