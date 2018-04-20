from __future__ import print_function
import sys
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabaz_score

if __name__ == '__main__':
    W = sparse.csr_matrix(np.load('W_{}.npy'.format(sys.argv[1])))
    print('Loaded W.')

    kmeans = MiniBatchKMeans(n_clusters=int(sys.argv[2]),
                             verbose=1,
                             random_state=1618,
                             max_no_improvement=None)
    kmeans.fit(W)
    print('Fit k-Means.')

    print('Cluster Centers: {}'.format(kmeans.cluster_centers_))
    np.save('labels.npy', kmeans.labels_)
    print('Inertia: {}'.format(kmeans.inertia_))
    print('Parameters: {}'.format(kmeans.get_params()))
    
    score = calinski_harabaz_score(W.toarray(), kmeans.labels_)

    print('Calinski-Harabaz Score: {}'.format(score))
    print('Success.')
