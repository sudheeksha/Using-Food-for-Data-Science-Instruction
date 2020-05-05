from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('/Users/sudheekshagarg/PycharmProjects/capstone/K-means&Agglomeration/HW_PCA_SHOPPING_CART_v892.csv')
df = df.drop(['ID'], axis=1)
print(df.head())
kmeans = KMeans(n_clusters=6).fit(df)
# print(kmeans.labels_)
list1 = kmeans.labels_

distortions = []
K = range(1,16)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df)
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

clustering = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward').fit(df)
list2 = clustering.labels_
Z = linkage(df, 'ward')
labelList = range(1, 11)

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=10,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    # leaf_font_size=12,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

print(list(set(list1) - set(list2)))
# print(clustering.labels_)