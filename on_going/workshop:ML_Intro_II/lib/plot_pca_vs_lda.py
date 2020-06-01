"""
=======================================================
Comparison of LDA and PCA 2D projection of Wine dataset
=======================================================

The Wine dataset represents 3 kind of Wine flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.
"""

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

wine = datasets.load_wine()

X = wine.data
y = wine.target
target_names = wine.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('PCA: explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print('LDA: explained variance ratio (first two components): %s'
      % str(lda.explained_variance_ratio_))

f, axes = plt.subplots(1, 2, figsize=(18, 6))
ax = axes[0]
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    ax.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
ax.legend()
ax.set_title('PCA of Wine dataset')

ax = axes[1]
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    ax.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
ax.legend()
ax.set_title('LDA of Wine dataset')

f.show()
