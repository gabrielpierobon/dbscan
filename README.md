# DBSCAN clustering for data shapes k-means can’t handle well (in Python)

In this post I’d like to take some content from Introduction to Machine Learning with Python by Andreas C. Müller & Sarah Guido and briefly expand on one of the examples provided to showcase some of the strengths of DBSCAN clustering when k-means clustering doesn’t seem to handle the data shape well. I’m going to go right to the point, so I encourage you to read the full content of Chapter 3, starting on page 168 if you would like to expand on this topic. I’ll be quoting the book when describing the working of the algorithm.

## Clustering
* it’s the task of partitioning the dataset into groups, called clusters
* the goal is to split up the data ins such a way that points within a single cluster are very similar and points in a different cluster are different k-means clustering tries to find cluster centers that are representative of certain regions of the data alternates between two steps: assigning each data point to the closest cluster center, and then setting each cluster center as the mean of the data points that are assigned to it the algorithm is finished when the assignment of instances to clusters no longer changes 

This is how k-means work in a visual representation:

```import mglearn
mglearn.plots.plot_kmeans_algorithm()```

One issue with k-means clustering is that it assumes that all directions are equally important for each cluster. This is usually not a big problem, unless we come across with some oddly shape data.

In this example, we will artificially generate that type of data. With the code below, provided by the authors of the book (with some minor changes in number of clusters), we can generate some data that k-means won’t be able to handle correctly:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate some random cluster data
X, y = make_blobs(random_state=170, n_samples=600, centers = 5)
rng = np.random.RandomState(74)
# transform the data to be stretched
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)
# plot
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

As you can see, we have arguably 5 defined clusters with a stretched diagonal shape.

Let’s apply k-means clustering:

# cluster the data into five clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_pred = kmeans.predict(X)
# plot the cluster assignments and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],   
            kmeans.cluster_centers_[:, 1],
            marker='^', 
            c=[0, 1, 2, 3, 4], 
            s=100, 
            linewidth=2,
            cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

What we can see here is that k-means has been able to correctly detect the clusters at the middle and bottom, while presenting trouble with the clusters at the top, which are very close to each other. The authors say: “these groups are stretched toward the diagonal. As k-means only considers the distance to the nearest cluster center, it can’t handle this kind of data”

Let’s see how DBSCAN clustering can help with this shape:

DBSCAN
Some highlights about DBSCAN clustering extracted from the book:

stands for “density based spatial clustering of applications with noise”
does not require the user to set the number of clusters a priori
can capture clusters of complex shapes
can identify points that are not part of any cluster (very useful as outliers detector)
is somewhat slower than agglomerative clustering and k-means, but still scales to relatively large datasets.
works by identifying points that are in crowded regions of the feature space, where many data points are close together (dense regions in feature space)
Points that are within a dense region are called core samples (or core points)
There are two parameters in DBSCAN: min_samples and eps
If there are at least min_samples many data points within a distance of eps to a given data point, that data point is classified as a core sample
core samples that are closer to each other than the distance eps are put into the same cluster by DBSCAN.
This is an example of how clustering changes according to the choosing of both parameters:

mglearn.plots.plot_dbscan()

In this plot, points that belong to clusters are solid, while the noise points are shown in white. Core samples are shown as large markers, while boundary points are displayed as smaller markers. Increasing eps (going from left to right in the figure) means that more points will be included in a cluster. This makes clusters grow, but might also lead to multiple clusters joining into one. Increasing min_samples (going from top to bottom in the figure) means that fewer points will be core points, and more points will be labeled as noise.
The parameter eps is somewhat more important, as it determines what it means for points to be close. Setting eps to be very small will mean that no points are core samples, and may lead to all points being labeled as noise. Setting eps to be very large will result in all points forming a single cluster.

Let’s get back to our example and see how DBSCAN deals with it:

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# cluster the data into five clusters
dbscan = DBSCAN(eps=0.123, min_samples = 2)
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

After twisting eps and min_samplesfor some time, I got some fairly consistent clusters, still including some noise points.

While DBSCAN doesn’t require setting the number of clusters explicitly, setting eps implicitly controls how many clusters will be found.
Finding a good setting for eps is sometimes easier after scaling the data, as using these scaling techniques will ensure that all features have similar ranges.
Lastly, considering we created the data points explicitly defining 5 clusters, we can mesure performance using adjusted_rand_score. This is not frequent since in real cases we don’t have cluster labels to begin with (thus our need to apply clustering techinques). Since in this case we do have labels, we can measure performance:

from sklearn.metrics.cluster import adjusted_rand_score
#k-means performance:
print("ARI =", adjusted_rand_score(y, y_pred).round(2))
ARI = 0.76
#DBSCAN performance:
print("ARI =", adjusted_rand_score(y, clusters).round(2))
ARI = 0.99
There you have it! DBSCAN scores 0.99 while k-means only gets 0.76
