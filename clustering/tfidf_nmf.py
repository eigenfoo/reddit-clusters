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
    # Helper function to mask out non-tokens
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
    # Strip HTML tags, strip links from markdown, and lowercase, respectively.
    document = re.sub(r'&[a-z]+;', r'', document)
    document = re.sub(r'\[(.+)\][(].+[)]', r'\1', document)
    document = document.lower()
    return document


def tokenize(document):
    # Tokenize by lemmatizing
    doc = nlp(document)
    return [token.lemma_ for token in doc if mask(token)]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ')
        print('\tpython tfidf_nmf.py SUBREDDIT_NAME NUM_TOPICS')
        sys.exit(1)

    # Disable tagger, parser and named-entity recognition
    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    # Load custom stoplist
    with open('../data/stoplist.txt', 'r') as f:
        stops = f.read().split()

    for stop in stops:
        STOP_WORDS.add(stop)

    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    # Read data.
    DATA_FILE = '../data/bigquery/2017/11-12/' + sys.argv[1] + '.csv'
    data = pd.read_csv(DATA_FILE)
    data = data.iloc[:, 1].fillna('').astype(str).squeeze()
    print('Loaded Reddit comments.')

    # Cut off the bottom 50% of all comments, by simple count of split tokens.
    counts = data.apply(lambda s: len(s.split()))
    threshold = counts.quantile(0.5)
    data = data[counts > threshold]
    print('High-pass filtered comments.')

    np.save('results/data_{}.npy'.format(sys.argv[1]), data)
    print('Saved high-pass filtered data.')

    # Vectorize data using tf-idfs.
    vectorizer = TfidfVectorizer(strip_accents='unicode',
                                 preprocessor=preprocess,
                                 tokenizer=tokenize,
                                 stop_words=list(STOP_WORDS),
                                 max_df=0.90,
                                 min_df=0.001,
                                 norm='l2')

    tfidf = vectorizer.fit_transform(data)
    print('Vectorized comments.')

    feature_names = vectorizer.get_feature_names()
    np.save('results/feature_names_{}.npy'.format(sys.argv[1]), feature_names)
    np.save('results/V_{}.npy'.format(sys.argv[1]), tfidf)

    print('Saved features names (vocabulary) and document-term matrix.')
    print('Factorizing tfidf matrix...')

    # Factorize with NMF.
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

    np.save('results/H_{}.npy'.format(sys.argv[1]), H)
    np.save('results/W_{}.npy'.format(sys.argv[1]), W)
    print('Saved factorization matrices.')

    print('')
    print('------------------------------')
    print('')

    print('Reconstruction error: {}'.format(err))
    print('')

    # FIXME This shouldn't be necessary? And yet it doesn't work without it :(
    data = np.load('data_{}.npy'.format(sys.argv[1]))

    # Print clusters and exemplars.
    for topic_idx, [scores, topic] in enumerate(zip(np.transpose(W), H)):
        print('Cluster #{}:'.format(topic_idx))
        print('Cluster importance: {}'.format(
            float((np.argmax(W, axis=1) == topic_idx).sum()) / W.shape[0]))

        for token, importance in zip(
                [feature_names[i] for i in np.argsort(topic)[:-15 - 1:-1]],
                np.sort(topic)[:-15 - 1:-1]):
            print('{}: {:2f}'.format(token, importance))

        print('')

        for exemplar_idx in np.argsort(scores)[-5:]:
            print(exemplar_idx)
            print(data[exemplar_idx])
            print('')

        print('----------')
