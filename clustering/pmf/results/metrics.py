import numpy as np
from sklearn.metrics import calinski_harabaz_score, davies_bouldin_score

# Evaluating PMF
print('NMF:')
U = np.load('U.npy')
V = np.load('V.npy')

print('Calinski-Harabaz:')
print(calinski_harabaz_score(V, np.argmax(V, axis=1)))  # 13.67495773754659
print(calinski_harabaz_score(U, np.argmax(U, axis=1)))  # 254.89565668100244

print('Davis-Bouldin:')
print(davies_bouldin_score(V, np.argmax(V, axis=1)))  # 1.422418647727864
print(davies_bouldin_score(U, np.argmax(U, axis=1)))  # 1.8371611775747891

print('')

# Evaluating NMF
print('PMF:')
H = np.transpose(np.load('../../nmf/results/H_NeutralPolitics.npy'))
W = np.load('../../nmf/results/W_NeutralPolitics.npy')

print('Calinski-Harabaz:')
print(calinski_harabaz_score(H, np.argmax(H, axis=1)))  # 143.55559101356215
print(calinski_harabaz_score(W, np.argmax(W, axis=1)))  # 431.88414916843504

print('Davis-Bouldin:')
print(davies_bouldin_score(H, np.argmax(H, axis=1)))  # 1.646735348262395
print(davies_bouldin_score(W, np.argmax(W, axis=1)))  # 0.9959409852151355
