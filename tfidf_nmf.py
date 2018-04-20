from __future__ import print_function
import sys
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def mask(token):
    if (not token.is_ascii
            or token.is_stop
            or token.is_space
            or token.like_url
            or token.like_num
            or token.like_email
            or token.is_punct
            or token.pos_ in ['X', 'SYM']):
        return False
    return True


def preprocess(document):
    document = re.sub(r"&[a-z]+;", r" ", document)
    document = document.lower()
    return document


def tokenize(document):
    doc = nlp(document)
    return [token.lemma_ for token in doc if mask(token)]


if __name__ == '__main__':
    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    with open('data/ranksnl_large.txt', 'r') as f:
        stops = f.read().split()

    for stop in stops:
        STOP_WORDS.add(stop)

    for word in STOP_WORDS:
        lexeme = nlp.vocab[unicode(word)]
        lexeme.is_stop = True

    DATA_FILE = './data/bigquery/' + sys.argv[1] + '.csv.gz'

    print('Loading Reddit comments...')

    data = pd.read_csv(DATA_FILE)
    data = data.loc[:, 'body'].fillna('').astype(str)

    print('Loaded Reddit comments.')
    print('Vectorizing comments...')

    vectorizer = TfidfVectorizer(strip_accents='unicode',
                                 preprocessor=preprocess,
                                 tokenizer=tokenize,
                                 stop_words=list(STOP_WORDS),
                                 max_df=0.90,
                                 min_df=0.001,
                                 norm=None)

    tfidf = vectorizer.fit_transform(data)

    # Remove bottom 20%. Save index as keep.
    norms = tfidf.sum(axis=1)
    threshold = np.percentile(norms, 20)
    keep = np.where(norms > threshold)[0]
    tfidf = tfidf[keep, :]
    tfidf = tfidf / tfidf.sum(axis=1)

    print('Vectorized comments.')

    feature_names = vectorizer.get_feature_names()
    np.save('feature_names_{}.npy'.format(sys.argv[1]), feature_names)
    np.save('tfidf_{}.npy'.format(sys.argv[1]), tfidf)
    np.save('keep_{}.npy'.format(sys.argv[1]), keep)

    print('Factorizing tfidf matrix...')

    # alpha controls strength of regularization
    # l1_ratio controls ratio of l1 and l2 regularization
    nmf = NMF(n_components=int(sys.argv[2]),
              init='nndsvd',
              max_iter=200,
              random_state=1618,
              alpha=0.2,
              l1_ratio=0,
              verbose=True)

    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    err = nmf.reconstruction_err_

    print('Factorized tfidf matrix.')

    np.save('H_{}.npy'.format(sys.argv[1]), H)
    np.save('W_{}.npy'.format(sys.argv[1]), W)

    print('Reconstruction error: {}'.format(err))

    print('')
    print('------------------------------')
    print('')

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join(['"' + feature_names[i] + '"'
                        for i in topic.argsort()[:-50 - 1:-1]]))
        print('')
