from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from scipy import linalg
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')  # needed for OSX
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

#################################################################################################
# Create dummy data                                                                             #
#################################################################################################
np.random.seed(17)
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
# iterate the algorithm until convergence                                                       #
#################################################################################################
for num_iter in range(11):

    algorithm = mixture.GaussianMixture(n_components=3, n_init=1, max_iter=num_iter+1, covariance_type='full', means_init=np.array([[0,-1],[2,-0.5],[1,2]]), init_params='random', random_state=1)
    algorithm.fit(X)
    fig, ax = plt.subplots()

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    plt.title(r"$" + str(num_iter) + "\mathrm{~iterations}$", size=24)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))

    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    # plot cluster centers
    centers = []
    ells = []
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    if hasattr(algorithm, 'means_'):
        centers = algorithm.means_
    if hasattr(algorithm, 'covariances_'):
        covars = algorithm.covariances_
        for i in range(len(covars)):
            v, w = linalg.eigh(covars[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            v, w = linalg.eigh(covars[i])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            ell = mpl.patches.Ellipse(centers[i], v[0], v[1], 180. + angle, facecolor=colors[i], edgecolor='black')
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    if centers != []:
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'], marker='X', alpha=0.6, linewidth=1.5, edgecolors='black')

    plt.savefig("./GMM-" + str(num_iter) + ".pdf", dpi=300, bbox_inches='tight')
    plt.close()
