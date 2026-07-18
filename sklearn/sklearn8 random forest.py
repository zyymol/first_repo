from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
X, y = load_digits(return_X_y=True)
n_trees = [1, 5, 10, 20, 50, 100, 200]
scores = []
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    s = cross_val_score(rf, X, y, cv=5).mean()
    scores.append(s)
    print(f'n={n:3d}: {s:.4f}')
print(f'\n单树 → 20棵树: 提升 {scores[3] - scores[0]:.4f}')
print(f'20棵 → 200棵:  提升 {scores[-1] - scores[3]:.4f}')
plt.figure(figsize=(8, 5))
plt.plot(n_trees, scores, 'o-', color='#639922', lw=2, markersize=8)
plt.xlabel('n_estimators'); plt.ylabel('CV accuracy')
plt.title('Random Forest: accuracy vs number of trees')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('rf_n_trees.png', dpi=150, bbox_inches='tight', facecolor='none')

#ex2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
X, y = load_digits(return_X_y=True)
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RF(n=20)': RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1),
    'RF(n=100)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LogisticReg': LogisticRegression(max_iter=5000),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5)
}

names=[]
means=[]
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5)
    names.append(name)
    means.append(s.mean())
    print(f'{name:20s}: {s.mean():.4f}')

plt.figure(figsize=(9, 5))
bars = plt.bar(names, means)
plt.ylabel('CV accuracy')
plt.title('5 Models on Digits (5-fold CV)')
# 标注数值
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', fontsize=10)
plt.ylim(0.75, 1.0)
plt.tight_layout()
plt.savefig('rf_vs_all.png', dpi=150, bbox_inches='tight', facecolor='none')

#ex3
import numpy as np
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
imp = rf.feature_importances_
top10 = np.argsort(imp)[-10:][::-1]
print('Top 10 most important pixels:')
for rank, idx in enumerate(top10, 1):
    row, col = divmod(idx, 8)
    print(f'  #{rank}: pixel({row},{col}), importance={imp[idx]:.4f}')