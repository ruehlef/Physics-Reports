"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
# print(__doc__)

import time
import warnings

import matplotlib as mpl
mpl.use('TkAgg')  # needed for OSX


import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(17)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 2000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': 0.3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
    (blobs, {}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    # (no_structure, {}),
    (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .17, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2})
    # (varied, {'eps': .18, 'n_neighbors': 2}),
    ]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    # connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    # connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    k_means = cluster.KMeans(n_clusters=params['n_clusters'])
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
    birch_clustering = cluster.Birch(n_clusters=params['n_clusters'])
    dbscan = cluster.DBSCAN(eps=params['eps'], min_samples=3)

    clustering_algorithms = (
        (r'\mathit{K}\!-\!means', k_means),
        (r'\mathit{K}\!-\!means~mini~batch', two_means),
        (r'Mean~shift', ms),
        (r'Gaussian~Expectation', gmm),
        (r'BIRCH', birch_clustering),
        (r'DBSCAN', dbscan)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(r"$\mathrm{" + name + "}$", size=24)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # plot cluster centers
        centers = []
        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
        if hasattr(algorithm, 'means_'):
            centers = algorithm.means_
        if hasattr(algorithm, 'subcluster_centers_'):
            sc_centers = algorithm.subcluster_centers_
            plt.scatter(sc_centers[:, 0], sc_centers[:, 1], s=100, c=['white','white','white','white','white','white','white','white','white'], marker='X', alpha=1, linewidth=1.5, edgecolors='black')

        if centers != []:
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'], marker='X', alpha=1, linewidth=1.5, edgecolors=['black', 'black','black','black','black','black','black','black','black'])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.95, .075, ('$%.2f\mathrm{s}$' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=24, horizontalalignment='right', backgroundcolor='white')
        plot_num += 1

# plt.show()
plt.savefig("./Comparison_clustering.pdf", dpi=300, bbox_inches='tight')
plt.close()

