"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""

# Lots of corrections are made from Init

import numpy as np
import scipy as sp
from scipy.special import gammaln
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

    def __init__(self, matrix, docs_edges, words, vocabulary, n_topics, lambda_param, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.lambda_param = lambda_param
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
        self._initialize(matrix)

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape
        self.matrix = np.copy(matrix)

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def _conditional_distribution(self, m, w, edge_dict):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / (self.nz + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * self.n_topics)
        topic_assignment = [0] * self.n_topics
        parent = self.nzw[:, w]
        try:
            edge_dict[w]
            C = self.phi()[:, edge_dict[w]]
            children = np.eye(C.shape[0])[C.argmax(0)]
            topic_assignment = children.sum(0)
            topic_assignment = topic_assignment / self.matrix[m, :].sum()
            topic_assignment = np.exp(np.dot(self.lambda_param, topic_assignment))
        except Exception as e:
            error_message = traceback.format_exc()
            if "edge_dict[w]" not in str(error_message):
                print(error_message)
            topic_assignment = np.exp(topic_assignment)
            topic_assignment = topic_assignment / self.matrix[m, :].sum()
            pass
        p_z = np.multiply(left, right) * topic_assignment
        return np.divide(p_z, p_z.sum())

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)
            
#             print(self.nzw[z,:])

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)
        
        for i in xrange(n_docs):
            count = 0
            edges_count = 0
            for a, b in (self.docs_edges[i]):
                edges_count += 1
                aa = self.nzw[:, a]
                bb = self.nzw[:, b]
                if aa.argmax() == bb.argmax():
                    count += 1
            if edges_count > 0:
                lik += np.log(np.exp(self.lambda_param*count/edges_count))

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=0)[np.newaxis, :]
        return num
    
    def theta(self):
        V = self.nmz.shape[1]
        num = self.nmz + self.alpha
        num /= np.sum(num, axis=1)[:, np.newaxis]
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
            worddict[t] = {}
            topWordIndices = pseudocounts[t, :].argsort()[-K:]
            worddict[t] = [vocab[i] for i in topWordIndices]
        return worddict

#     def getTopKWords(self, K, vocab):
#         """
#         Returns top K discriminative words for topic t v for which p(v | t) is maximum
#         """
#         pseudocounts = np.copy(self.nzw.T)
#         normalizer = np.sum(pseudocounts, (0))
#         pseudocounts /= normalizer[np.newaxis, :]
#         worddict = {}
#         for t in range(self.n_topics):
#             worddict[t] = {}
#             topWordIndices = pseudocounts[:, t].argsort()[-(K+1):-1]
#             worddict[t] = [vocab[i] for i in topWordIndices]
#         return worddict

    def run(self, maxiter=100):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = self.matrix.shape

        for it in xrange(maxiter):
            print "Iteration", it
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(self.matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w, self.edge_dict[m])
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z
            self.likelihood_history.append(self.loglikelihood())