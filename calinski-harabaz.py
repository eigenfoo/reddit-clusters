from __future__ import print_function
import sys
import numpy as np
from sklearn.metrics import calinski_harabaz_score

if __name__ == '__main__':
    with open('scores.txt', 'w') as f:
        for n in range(8, 31):
            W = np.load('W_{}.npy'.format(n))
            classes = np.zeros(W.shape[0])
            for idx, doc in enumerate(W):
                classes[idx] = np.argmax(doc)
            score = calinski_harabaz_score(W, classes)
            print("{},{}".format(n, score), file=f)
