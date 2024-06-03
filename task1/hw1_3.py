from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=500, # 500个样本
                 n_features=2, # 每个样本2个特征
                 centers=4, # 4个中心
                 random_state=1 #控制随机性
                 )

# color = ['red', 'pink','orange','gray']
# fig, axi1=plt.subplots(1)
# for i in range(4):
#     axi1.scatter(X[y==i, 0], X[y==i,1],
#                marker='o',
#                s=8,
#                c=color[i]
#                )
# plt.show()

n_clusters=3
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)

centroid=cluster.cluster_centers_

y_pred = cluster.labels_#获取训练后对象的每个样本的标签    
color=['red','pink','orange','gray']
fig, axi1=plt.subplots(1)
for i in range(n_clusters):
    axi1.scatter(X[y_pred==i, 0], X[y_pred==i, 1],
               marker='o',
               s=8,
               c=color[i])
axi1.scatter(centroid[:,0],centroid[:,1],marker='x',s=100,c='black')

# n_clusters=4
# cluster2 = KMeans(n_clusters=n_clusters,random_state=0).fit(X)

# centroid=cluster2.cluster_centers_

# y_pred = cluster2.labels_#获取训练后对象的每个样本的标签    
# centtrod = cluster2.cluster_centers_
# color=['red','pink','orange','gray']
# fig, axi1=plt.subplots(1)
# for i in range(n_clusters):
#     axi1.scatter(X[y_pred==i, 0], X[y_pred==i, 1],
#                marker='o',
#                s=8,
#                c=color[i])
# axi1.scatter(centroid[:,0],centroid[:,1],marker='x',s=100,c='black')