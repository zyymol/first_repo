import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#ex1
X, y = make_blobs(n_samples=500, centers=2, random_state=42, cluster_std=2.0)
Xs = StandardScaler().fit_transform(X)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, C in zip(axes, [0.01, 1.0, 100.0]):
    lr = LogisticRegression(C=C, max_iter=5000).fit(Xs, y)
    # 构造网格
    xx, yy = np.meshgrid(
        np.linspace(Xs[:, 0].min()-0.5, Xs[:, 0].max()+0.5, 300),
        np.linspace(Xs[:, 1].min()-0.5, Xs[:, 1].max()+0.5, 300)
    )
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # 画决策区域 + 数据点
    ax.contourf(xx, yy, Z, alpha=0.1, cmap='coolwarm')
    ax.scatter(Xs[:, 0][y==0], Xs[:, 1][y==0], c='#378ADD', s=20, label='class 0')
    ax.scatter(Xs[:, 0][y==1], Xs[:, 1][y==1], c='#D85A30', s=20, label='class 1')
    ax.set_title(f'C = {C}', fontsize=14)
    ax.legend()
fig.suptitle('Decision boundary vs C (regularization)')
fig.tight_layout()
plt.savefig('lr_decision_boundary.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X, y = digits.data, (digits.target == 5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te  = scaler.transform(X_test)
lr = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr, y_train)
print(f'Accuracy: {lr.score(X_te, y_test):.4f}')
print(f'正类(5)概率均值: {lr.predict_proba(X_te)[:, 1][y_test == 1].mean():.3f}')
print()
# 系数解读：绝对值最大的10个像素位置
top10 = np.argsort(np.abs(lr.coef_[0]))[-10:][::-1]
print('最重要10个像素（位置, 系数）:')
for idx in top10:
    row, col = divmod(idx, 8)
    direction = '提升5的概率' if lr.coef_[0][idx] > 0 else '降低5的概率'
    print(f'  像素({row},{col}): {lr.coef_[0][idx]:+.4f} → {direction}')
#ex1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
X, y = load_digits(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
# 多分类 LR（自动 one-vs-rest）
lr_scores = cross_val_score(LogisticRegression(max_iter=5000), Xs, y, cv=5)
# KNN
knn_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), Xs, y, cv=5)

print(f'LogisticRegression: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})')
print(f'KNN(k=5):          {knn_scores.mean():.4f} (+/- {knn_scores.std():.4f})')
print(f'差异: {lr_scores.mean() - knn_scores.mean():.4f}')