from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np
#ex1
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
Xs = StandardScaler().fit_transform(X)

km1 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(Xs)
db1 = DBSCAN(eps=0.3, min_samples=5).fit(Xs)
n_noise = (db1.labels_ == -1).sum()
print(f'n_noise={n_noise:3d}')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(Xs[:,0], Xs[:,1], c=km1.labels_, cmap='tab10', s=15)
axes[0].set_title(f'KMeans (K=2)')
axes[1].scatter(Xs[:,0], Xs[:,1], c=db1.labels_, cmap='tab10', s=15)
axes[1].set_title(f'DBSCAN (eps=0.3), {n_noise} noise points')
fig.suptitle('KMeans vs DBSCAN on Moons')
fig.tight_layout()
plt.savefig('15_dbscan_vs_kmeans.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex2
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
Xs = StandardScaler().fit_transform(X)
km2 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
agg = AgglomerativeClustering(n_clusters=3).fit(Xs)
db2 = DBSCAN(eps=0.5, min_samples=5).fit(Xs)
print('KMeans:         ', km2.labels_[:10].tolist())
print('Agglomerative:  ', agg.labels_[:10].tolist())
print('DBSCAN:         ', db2.labels_[:10].tolist())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(Xs[:,0][:10], Xs[:,1][:10], c=km2.labels_[:10], cmap='tab10', s=15)
axes[0].set_title(f'KMeans (K=2)', fontsize=13)
axes[1].scatter(Xs[:,0][:10], Xs[:,1][:10], c=agg.labels_[:10], cmap='tab10', s=15)
axes[1].set_title(f'Agglomerative (n_clusters=3)', fontsize=13)
axes[2].scatter(Xs[:,0][:10], Xs[:,1][:10], c=db2.labels_[:10], cmap='tab10', s=15)
axes[2].set_title(f'DBSCAN (eps=0.5)', fontsize=13)
fig.suptitle('KMeans vs DBSCAN on Moons', fontsize=15)
fig.tight_layout()
plt.savefig('15_dbscan_vs_km_vs_agg.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
Xs = StandardScaler().fit_transform(X)
for i in [0.2,0.3,0.5]:
    db3 = DBSCAN(eps=i, min_samples=5).fit(Xs)
    n_noise = (db3.labels_ == -1).sum()
    print(f'n_noise={n_noise:3d} when eps={i}')#随eps递减，0.5时没有分到noise的