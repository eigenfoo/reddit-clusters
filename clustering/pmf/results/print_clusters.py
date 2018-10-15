from __future__ import print_function
import numpy as np

R = np.load('R.npy')
U = np.load('U.npy')
V = np.load('V.npy')
feature_names = np.load('feature_names.npy')

# Print clusters and exemplars.
for topic_idx, [_, token_scores] in enumerate(zip(np.transpose(U),
                                                  np.transpose(V))):
    print('Cluster #{}:'.format(topic_idx))
    print('Cluster importance: {}'.format(
        float((np.argmax(U, axis=1) == topic_idx).sum()) / U.shape[0]))

    for token, importance in zip(
            [feature_names[i] for i in np.argsort(token_scores)[:-15 - 1:-1]],
            np.sort(token_scores)[:-15 - 1:-1]):
        print('{}: {:2f}'.format(token, importance))

    print('')
    print('----------')
