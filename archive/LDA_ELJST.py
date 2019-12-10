"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""
# Called EJLST - notest

import numpy as np
import datetime
import scipy as sp
from scipy.special import gammaln
from scipy import sparse
import traceback

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, matrix, sentiment, docs_edges, words, vocabulary, n_topics, n_sentiment, lambda_param=1.0, alpha=0.1, beta=0.1, gamma = 0.5, SentimentRange=5):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.n_sentiment = n_sentiment
        self.alpha = alpha
        self.beta = beta
        self.gamma = 10.0/(n_topics * n_sentiment)
        self.gammavec = None
        self.lambda_param = lambda_param
        self.SentimentRange = SentimentRange
        self.probabilities_ts = {}
        self.sentimentprior = {}
        self.sentiment = None
        self._initialize(matrix, sentiment)
        self.words = words
        self.vocabulary = vocabulary
        
        edge_dict__ = []
        for doc in docs_edges:
            edge_dict_ = {}
            for i, j in doc:
                try:
                    edge_dict_[i] += [j]
                except:
                    edge_dict_[i] = [j]
                try:
                    edge_dict_[j] += [i]
                except:
                    edge_dict_[j] = [i]
            edge_dict__.append(edge_dict_)
            
        self.edge_dict = edge_dict__
        self.docs_edges = docs_edges
        self.likelihood_history = []

    def _initialize(self, matrix, sentiment):

        n_docs, vocab_size = matrix.shape
        
        self.matrix = np.copy(matrix)
        self.sentiment = sentiment.copy()

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        self.nmzs = np.zeros((n_docs, self.n_topics, self.n_sentiment))
        self.nm = np.zeros(n_docs)
        self.nzws = np.zeros((self.n_topics, vocab_size, self.n_sentiment))
        self.nzs = np.zeros((self.n_topics, self.n_sentiment))
        self.topics = {}
        self.sentiments = {}
        
        self.gammavec = []
        for i in sentiment:
            p = [self.gamma] * self.n_sentiment
            p[int(i)-1] += 1
            self.gammavec.append(p)
        
#         for _ in range(len(test_matrix)):
#             self.gammavec.append(self.gamma * np.ones(self.n_sentiment))
            
        self.gammavec = np.array(self.gammavec)

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                s = np.random.randint(self.n_sentiment)
                self.nmz[m,z] += 1
                self.nmzs[m,z, s] += 1
                self.nm[m] += 1
                self.nzws[z,w, s] += 1
                self.nzs[z, s] += 1
                self.topics[(m,i)] = z
                self.sentiments[(m,i)] = s

    def _conditional_distribution(self, m, w, edge_dict):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzws.shape[1]
        left = (self.nzws[:, w, :] + self.beta) / (self.nzs + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * self.n_topics)
        gammaFactor = ((self.nmzs[m, :, :] + self.gammavec[m]).T/(self.nmz[m, :] + np.sum(self.gammavec[m]))).T
        
        topic_assignment = np.ones(self.nzws[:, 0, :].shape)
        try:
            edge_dict[w]
            parent = self.nzws[:, w , :].sum(-1)
            new_C = np.zeros((self.phi()[:, edge_dict[w], :].shape))
            for idx, i in enumerate(edge_dict[w]):
                C = self.phi()[:, i, :]
                C[np.where(C == C.max())] = 1
                C[np.where(C != C.max())] = 0
                new_C[:, idx, :] = C

            topic_assignment = new_C.sum(1)
            topic_assignment /= topic_assignment.sum()
            topic_assignment = np.exp(self.lambda_param * topic_assignment)
        except Exception as e:
            error_message = traceback.format_exc()
            if "edge_dict[w]" not in str(error_message):
                print(error_message)
            pass

        p_zs = left * right[:, np.newaxis] * gammaFactor * topic_assignment
        p_zs /= np.sum(p_zs)
        return p_zs

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzws.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            for s in xrange(self.n_sentiment):
                lik += log_multi_beta(self.nzws[z, :, s]+self.beta)
                lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            for z in xrange(self.n_topics):
                lik += log_multi_beta(self.nmzs[m, z, :]+self.gammavec[m])
                lik -= log_multi_beta(self.gammavec[m], None)
        
        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)
        
        for i in xrange(n_docs):
            for s in xrange(self.n_sentiment):
                count = 0
                edges_count = 0
#                 print(self.nzws.shape)
                for a, b in (self.docs_edges[i]):
                    edges_count += 1
                    aa = self.nzws[:, a, s]
                    bb = self.nzws[:, b, s]
                    if aa.argmax() == bb.argmax():
                        count += 1
                if edges_count > 0:
                    lik += np.log(np.exp(self.lambda_param*count/edges_count))

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        num = self.nzws + self.beta
        n = np.sum(num, axis=1)
        n = n[:, np.newaxis, :]
        num /= n
        return num
    
    def theta(self):
        num = self.nmz + self.alpha
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num
    
    def pi(self):
        num = self.nmzs + self.gammavec[:, np.newaxis, :]
        n = np.sum(num, axis=2)
        n = n[: ,:, np.newaxis]
        num /= n
        return num
    
    def getTopKWords(self, K, vocab):
        """
        Returns top K discriminative words for topic t v for which p(v | t) is maximum
        """
        pseudocounts = self.phi().copy() #np.copy(self.nzws)
        #normalizer = np.sum(pseudocounts, axis = 2)
        #normalizer = np.sum(normalizer, axis = 0)
        #pseudocounts /= normalizer[np.newaxis, :,  np.newaxis]
        worddict = {}
        for t in range(self.n_topics):
            for s in range(self.n_sentiment):
                worddict[(t, s)] = {}
                topWordIndices = pseudocounts[t, :, s].argsort()[-K:]
                worddict[(t, s)] = [vocab[i] for i in topWordIndices]
        return worddict

    def run(self, maxiter=20):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = self.matrix.shape
        
        for it in xrange(maxiter):
            print "Iteration", it
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(self.matrix[m, :])):
                    z = self.topics[(m,i)]
                    s = self.sentiments[(m,i)]
                    self.nmz[m, z] -= 1
                    self.nmzs[m, z, s] -= 1
                    self.nm[m] -= 1
                    self.nzws[z, w, s] -= 1
                    self.nzs[z, s] -= 1

                    p_z = self._conditional_distribution(m, w, self.edge_dict[m])
                    ind = sample_index(p_z.flatten())
                    
                    z, s = np.unravel_index(ind, p_z.shape)
                    
                    self.nmz[m, z] += 1
                    self.nmzs[m, z, s] += 1
                    self.nm[m] += 1
                    self.nzws[z, w, s] += 1
                    self.nzs[z, s] += 1
                    
                    self.topics[(m,i)] = z
                    self.sentiments[(m,i)] = s
            self.likelihood_history.append(self.loglikelihood())