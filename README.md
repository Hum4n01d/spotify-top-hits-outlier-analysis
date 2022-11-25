# Top Spotify Songs (2010-2019) Outlier Analysis

We hope to be able to identify hit songs that stand out from the other popular songs. Then, we could possibly generalize this model to personal user playlists in order to help them identify their own songs that are different from the others. 

## Data Preprocessing

Since the most of the data does not seem to be normal, we will be using normalization to scale our quantitative data. Then, we will apply PCA in order to reduce the number of dimensions of our data and simplify it. Finally, we will use value replacement for our categorical data.

## Building First Model

Since K means does not allow us to incorporate the categorical data, we used a variation included in the 'kmeans' package called K-Prototypes that can handle both numerical and categorical data. In order to identify the outliers, we calculated the cost of the points with respect to the centroid of its cluster (the cluster in which it belonged to) and we identified the top 10 points that were the furthest away, which were the outliers. 
