from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import numpy as np

# imports and definitions for plotting
import matplotlib as mpl
mpl.use('TkAgg')  # needed for OSX
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

#################################################################################################
# Create dummy data                                                                             #
#################################################################################################
np.random.seed(0)
n_samples = 700
blob = datasets.make_blobs(n_samples=n_samples, centers=3, random_state=8, cluster_std=1.3)

#################################################################################################
# Set up figures for plotting results                                                           #
#################################################################################################
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plt.close()

#################################################################################################
# Normalize data                                                                                #
#################################################################################################
dataset = blob
X, y = dataset

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)
X = np.vstack((X, np.array([[0, 1], [-0.5, 1], [2, 0]])))

#################################################################################################
# iterate the algorithm until convergence (three times)                                         #
#################################################################################################

for num_iter in range(0, 5):

    algorithm = cluster.KMeans(n_init=1, n_clusters=3, max_iter=num_iter+1, random_state=1, init=np.array([[-2,2],[-1,1],[1,-1]]))
    algorithm.fit(X)

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    # coloring scheme for clusters
    colors = np.array(list(islice(
        cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']),
        int(max(y_pred) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    # plot the points in the color corresponding to the cluster to which they are assigned
    plt.title(r"$" + str(num_iter) + "\mathrm{~iterations}$", size=24)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    # plot cluster centers
    centers = []
    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
    if hasattr(algorithm, 'means_'):
        centers = algorithm.means_
    if hasattr(algorithm, 'subcluster_centers_'):
        sc_centers = algorithm.subcluster_centers_
        plt.scatter(sc_centers[:, 0], sc_centers[:, 1], s=100, c=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'], marker='X', alpha=1, linewidth=1.5, edgecolors='black')

    if centers != []:
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'], marker='X', alpha=0.6, linewidth=1.5, edgecolors='black')

    plt.savefig("./K-Means" + str(num_iter) + ".pdf", dpi=300, bbox_inches='tight')
    plt.close()
