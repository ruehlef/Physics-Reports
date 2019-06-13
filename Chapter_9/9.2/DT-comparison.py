from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn import tree
import graphviz  # needed for plotting the trees

import matplotlib as mpl
mpl.use('TkAgg')  # needed for OSX
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from itertools import cycle, islice


#################################################################################################
# Create dummy data set with three classes in 2D feature space                                  #
#################################################################################################
n_classes = 3
X, y = make_blobs(n_samples=100, n_features=2, centers=n_classes, cluster_std=0.6, random_state=0)
plt.title(r"$\mathrm{Input~data}$", size=24)
feature_names = ["x1", "x2"]
cluster_names = ["c1", "c2", "c3"]

#################################################################################################
# Plot the data set                                                                             #
#################################################################################################
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']), int(max(y) + 1))))
cm = LinearSegmentedColormap.from_list("my_cmap", colors, N=len(colors))

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], s=25, color=colors[y])
plt.savefig("./DT_DummyData.pdf", dpi=300, bbox_inches='tight')
# plt.show()

#################################################################################################
# Perform a train:test split                                                                    #
#################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#################################################################################################
# CART                                                                                          #
#################################################################################################
clf = DecisionTreeClassifier(criterion='gini')  # use gini for splitting
clf.fit(X_train, y_train)  # fit the tree with the train set
y_pred = clf.predict(X_test)  # get the predictions for the test set
print("Accuracy CART: ", accuracy_score(y_test, y_pred))
# plot the tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=cluster_names, filled=False, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("DT_CART")

plt.clf()
plot_step = 0.02

# generate the regions and decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cm, alpha=0.7)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)

#################################################################################################
# Plot the points in the corresponding cluster color                                            #
#################################################################################################
for i, color in zip(range(n_classes), colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=cluster_names[i], cmap=colors, edgecolor='black', s=25)

plt.suptitle(r"$\mathrm{Decision~surface~of~the~CART~tree}$", size=24)
plt.axis("tight")
plt.savefig("./DT_Decision_boundaries_CART.pdf", dpi=300, bbox_inches='tight')
# plt.show()


#################################################################################################
# ID3                                                                                           #
#################################################################################################
clf = DecisionTreeClassifier(criterion='entropy')  # use entropy for splitting
clf.fit(X_train, y_train)  # fit the tree with the train set
y_pred = clf.predict(X_test)   # get the predictions for the test set
print("Accuracy ID3: ", accuracy_score(y_test, y_pred))
# plot the tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=cluster_names, filled=False, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("DT_ID3")


#################################################################################################
# Random Forest model                                                                           #
#################################################################################################
clf = RandomForestClassifier(n_estimators=3, max_features="auto", random_state=0)  # use three trees
clf.fit(X_train, y_train)  # fit the tree with the train set
y_pred = clf.predict(X_test)  # get the predictions for the test set
print("Accuracy Random Forest: ", accuracy_score(y_test, y_pred))
# plot the forest: iterate over the trees in the forest
for i in range(len(clf.estimators_)):
    dot_data = tree.export_graphviz(clf.estimators_[i], out_file=None, feature_names=feature_names, class_names=cluster_names, filled=False, rounded=True, special_characters=True, leaves_parallel=True)
    graph = graphviz.Source(dot_data)
    graph.render("RF_"+str(i))

#################################################################################################
# AdaBoost model                                                                                #
#################################################################################################
clf = AdaBoostClassifier(n_estimators=3)  # use three consecutive estimators
clf.fit(X_train, y_train)  # fit the tree with the train set
y_pred = clf.predict(X_test)  # get the predictions for the test set
print("Accuracy AdaBoost: ", accuracy_score(y_test, y_pred))
# print the tree weights and errors
print("Tree weights:", clf.estimator_weights_)
print("Tree errors:", clf.estimator_errors_)

# plot the trees: iterate over the consecutive trees
for i in range(len(clf.estimators_)):
    dot_data = tree.export_graphviz(clf.estimators_[i], out_file=None, feature_names=feature_names, class_names=cluster_names, filled=False, rounded=True, special_characters=True, leaves_parallel=True)
    graph = graphviz.Source(dot_data)
    graph.render("ADABoost_"+str(i))


#################################################################################################
# Gradient Boost model                                                                          #
#################################################################################################
clf = GradientBoostingClassifier(n_estimators=1, max_depth=1, criterion="mse")  # split on MSE
clf.fit(X_train, y_train)  # fit the tree with the train set
y_pred = clf.predict(X_test)  # get the predictions for the test set
print("Accuracy Gradient Boost: ", accuracy_score(y_test, y_pred))

# loop over estimators
for i in range(len(clf.estimators_)):
    # loop over trees for each estimator
    for j in range(len(clf.estimators_[i])):
        dot_data = tree.export_graphviz(clf.estimators_[i, j], out_file=None, feature_names=feature_names, class_names=cluster_names, filled=False, rounded=True, special_characters=True, leaves_parallel=True)
        graph = graphviz.Source(dot_data)
        graph.render("GBoost_"+ str(i)+"_"+ str(j))

#################################################################################################
# Optional: Crop all pdfs                                                                       #
#################################################################################################
from subprocess import call
import glob
pdfs = glob.glob("*.pdf")
for s in pdfs:
    call('pdfcrop ' + s, shell=True)
    call('rm -rf ' + s, shell=True)
    call('rm -rf ' + s[:-4], shell=True)
    call('mv ' + s[:-4] + '-crop.pdf ' + s, shell=True)
