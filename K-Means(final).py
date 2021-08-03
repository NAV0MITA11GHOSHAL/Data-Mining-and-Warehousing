%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(9, 9))

X, y = make_blobs(n_samples = 400, centers = 3, cluster_std = 0.63, random_state = 0)
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 18, cmap = 'spring')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()
