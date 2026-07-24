from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
#ex1
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xs)
y_pred = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(Xs[:,0], Xs[:,1], c=y_true, cmap='tab10', s=15)#cmap指颜色映射，tab10区分明显
axes[0].set_title('True labels')
axes[1].scatter(Xs[:,0], Xs[:,1], c=y_pred, cmap='tab10', s=15)
#axes[1].scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],c='red', marker='x', s=100, linewidths=2)#标中心点
axes[1].set_title('KMeans clusters (K=4)')
fig.suptitle('K-Means Comparison')
fig.tight_layout()
plt.savefig('14_kmeans_clusters.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex2
inertias=[]
for k in range(1, 9):
    kmi = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(kmi.inertia_)
#print(inertias)
plt.figure(figsize=(10, 8))
plt.plot(range(1, 9), inertias, 'o-', color='#378ADD', lw=2, markersize=8)
plt.axvline(4, color='#D85A30', ls='--', label='elbow: K=4')
plt.xlabel('n_clusters (K)')
plt.ylabel('inertia')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('14_kmeans_elbow.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
print('True labels  (first 20):', y_true[:20].tolist())
print('KMeans labels (first 20):', km.labels_[:20].tolist())
