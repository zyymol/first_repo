from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from sklearn.linear_model import Lasso

#ex1: Ridge压缩系数
X, y = load_diabetes(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
lr = LinearRegression().fit(Xs, y)
print(f'LR max |coef|: {abs(lr.coef_).max():.1f}')#.coef_是系数的数组
for alpha in [0.01, 0.1, 1, 10, 100]:#0.01系数max abs =37.6，100时21.4
    ridge = Ridge(alpha=alpha).fit(Xs, y)
    nz = np.sum(np.abs(ridge.coef_) > 1e-6)
    print(f'alpha={alpha:6.1f}: max|coef|={abs(ridge.coef_).max():.1f}, all {nz} non-zero')
#ex2
for alpha in [0.01, 0.1, 1, 10, 100]:#0.01剩10个完整，1剩7个
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(Xs, y)
    nz = np.sum(np.abs(lasso.coef_) > 1e-6)
    print(f'alpha={alpha:6.2f}: R²={lasso.score(Xs, y):.4f}, non-zero={nz}/10')
#ex3
import matplotlib.pyplot as plt
models={
    'lr':LinearRegression().fit(Xs, y),
    'ridge1':Ridge(alpha=1).fit(Xs, y),
    'lasso1':Lasso(alpha=1,max_iter=10000).fit(Xs, y),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, coef) in zip(axes, models.items()):
    colors = ['blue' if c > 0 else 'red' for c in coef.coef_]
    ax.bar(range(10), coef.coef_, color=colors)
    ax.grid(alpha=0.5)
fig.suptitle('LR vs Ridge vs Lasso', fontsize=15)
fig.tight_layout()

plt.savefig('12_lr_ridge_lasso_vs.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()