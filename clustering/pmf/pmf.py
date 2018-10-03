'''
Probabilisitic Matrix Factorization (PMF) using Tensorflow.

Original paper:
    http://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(1618)
tf.set_random_seed(1618)

DATA_FILE = 'NeutralPolitics.csv'
NUM_TOKENS = 5000
NUM_DOCUMENTS = 100
NUM_LATENTS = 20
NUM_EPOCHS = 2000


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


def variable_summaries(name, var):
    """ Attach summaries to a Tensor (for TensorBoard visualization). """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        minimum = tf.reduce_min(var)
        maximum = tf.reduce_max(var)

        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', maximum)
        tf.summary.scalar('min', minimum)
        tf.summary.histogram('histogram', var)


# Disable tagger, parser and named-entity recognition
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

# Read data
data = pd.read_csv(DATA_FILE).squeeze()

# Vectorize data with tf-idfs
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             tokenizer=tokenize,
                             max_df=0.90,
                             min_df=0.001,
                             norm='l2')

tfidf = vectorizer.fit_transform(data)

# Get portions of tf-idf matrix that are nonzero
nonzero_rows, nonzero_cols, nonzero_vals = sparse.find(tfidf)
index = np.vstack([nonzero_rows, nonzero_cols]).T
nonzero_tfidf = tfidf[nonzero_rows, nonzero_cols]

# Define matrices
with tf.name_scope('matrices'):
    U = tf.get_variable('U', [NUM_LATENTS, NUM_DOCUMENTS], tf.float32,
                        tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
    V = tf.get_variable('V', [NUM_LATENTS, NUM_TOKENS], tf.float32,
                        tf.truncated_normal_initializer(mean=0.0, stddev=0.2))
    R = tf.matmul(tf.transpose(U), V)

variable_summaries('U', U)
variable_summaries('V', V)
variable_summaries('R', R)

# Define loss
with tf.name_scope('loss'):
    # TODO regularization parameters may need tuning...
    lambda_U = 1 / (NUM_LATENTS * NUM_DOCUMENTS)
    lambda_V = 1 / (NUM_LATENTS * NUM_TOKENS)
    error = tf.reduce_sum((nonzero_tfidf - tf.gather_nd(R, index))**2)
    regularization_U = lambda_U * tf.reduce_sum(tf.norm(U, axis=1))
    regularization_V = lambda_V * tf.reduce_sum(tf.norm(V, axis=1))
    loss = error + regularization_U + regularization_V

    tf.summary.scalar('error', error)
    tf.summary.scalar('regularization_U', regularization_U)
    tf.summary.scalar('regularization_V', regularization_V)
    tf.summary.scalar('loss', loss)

# Define training
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(NUM_EPOCHS):
    _, summary_, loss_ = sess.run([train_step, merged_summary, loss])
    writer.add_summary(summary_, i)

    if i % 500 == 0:
        print('Epoch {}:'.format(i), loss_)

writer.close()
