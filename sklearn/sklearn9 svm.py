import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#ex1
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
Xs = StandardScaler().fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, kernel in zip(axes, ['linear', 'rbf']):
    svc = SVC(kernel=kernel, random_state=42).fit(Xs, y)
    xx, yy = np.meshgrid(
        np.linspace(Xs[:,0].min()-0.5, Xs[:,0].max()+0.5, 300),
        np.linspace(Xs[:,1].min()-0.5, Xs[:,1].max()+0.5, 300)
    )
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm')
    ax.scatter(Xs[:,0][y==0], Xs[:,1][y==0], c='#378ADD', s=15)
    ax.scatter(Xs[:,0][y==1], Xs[:,1][y==1], c='#D85A30', s=15)
    ax.set_title(f'kernel={kernel}, accuracy={svc.score(Xs, y):.3f}', fontsize=13)
fig.suptitle('SVM on make_moons: linear vs RBF kernel', fontsize=15)
fig.tight_layout()
plt.savefig('svc_kernels.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex2
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
X, y = load_digits(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
for k in ['linear', 'rbf', 'poly']:
    svc = SVC(kernel=k, random_state=42)
    s = cross_val_score(svc, Xs, y, cv=5)
    print(f'{k:6s}: {s.mean():.4f} (+/- {s.std():.4f})')
#ex3
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
models = {
    'DT': DecisionTreeClassifier(random_state=42),
    'RF(100)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LR': LogisticRegression(max_iter=5000),
    'SVC(rbf)': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
names=[]
means=[]
for name, m in models.items():
    s = cross_val_score(m, Xs, y, cv=5)
    names.append(name)
    means.append(s.mean())
order = np.argsort(means)[::-1]
names = [names[i] for i in order]
means = [means[i] for i in order]
plt.figure(figsize=(9, 5))
bars = plt.bar(names, means, color=['#7F77DD','#639922','#378ADD','#D85A30','#888780'])
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center')
plt.ylabel('CV accuracy'); plt.ylim(0.75, 1.0)
plt.title('All Models on Digits (5-fold CV)')
plt.tight_layout()
plt.savefig('svc_vs_all.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()