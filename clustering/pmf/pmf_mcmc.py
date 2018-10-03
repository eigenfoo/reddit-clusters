import numpy as np
import pandas as pd
import spacy
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pymc3 as pm
import theano
import theano.tensor as tt
import pickle


def mask(token):
    # Helper function to mask out non-tokens
    if (not token.is_ascii
            or token.is_stop
            or token.like_num
            or token.pos_ in ['X', 'SYM']):
        return False
    return True


def tokenize(document):
    # Tokenize by lemmatizing
    doc = nlp(document)
    return [token.lemma_ for token in doc if mask(token)]


# Disable tagger, parser and named-entity recognition
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

# Read data
DATA_FILE = 'NeutralPolitics.csv'
data = pd.read_csv(DATA_FILE).squeeze()

# Vectorize data using tf-idfs
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             tokenizer=tokenize,
                             max_df=0.90,
                             min_df=0.001,
                             norm='l2')

tfidf = vectorizer.fit_transform(data)
feature_names = vectorizer.get_feature_names()


def sparse_std(x, axis=None):
    """ Standard deviation of a scipy.sparse matrix, via [E(X^2) - E(X)^2]^(1/2) """
    return np.sqrt(np.mean(x.power(2), axis=axis) - np.square(np.mean(x, axis=axis)))


rows, columns, entries = scipy.sparse.find(tfidf)

n, m = tfidf.shape
dim = 20

sigma = entries.std()
sigma_u = sparse_std(tfidf, axis=1).mean()
sigma_v = sparse_std(tfidf, axis=0).mean()

with pm.Model() as pmf:
    U = pm.Normal('U', mu=0, sd=sigma_u, shape=[n, dim])
    V = pm.Normal('V', mu=0, sd=sigma_v, shape=[m, dim])
    R_nonzero = pm.Normal('R_nonzero',
                          mu=tt.sum(np.multiply(U[rows, :], V[columns, :]), axis=1),
                          sd=sigma,
                          observed=entries)
    trace = pm.sample()


with open('pmf_model.pkl', 'wb') as buff:
    pickle.dump({'model': pmf, 'trace': trace}, buff)
