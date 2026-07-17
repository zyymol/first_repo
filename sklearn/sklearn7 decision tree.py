from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
X, y = load_digits(return_X_y=True)
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
print(f'节点总数: {dt.tree_.node_count}')
print(f'实际深度: {dt.get_depth()}')
print(f'特征重要性(前5): {dt.feature_importances_[:5].round(4)}')
plt.figure(figsize=(14, 8))
plot_tree(dt, filled=True, feature_names=[f'p{x}' for x in range(64)],
          class_names=[str(i) for i in range(10)], fontsize=7)
plt.title('Decision Tree (max_depth=3) on Digits')
plt.tight_layout()
plt.savefig('dt_viz.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex2
from sklearn.model_selection import cross_val_score
import numpy as np
depths = range(1, 21)
scores = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    s = cross_val_score(dt, X, y, cv=5).mean()
    scores.append(s)
best_d = depths[np.argmax(scores)]#最大值所在索引
print(f'最佳 max_depth: {best_d}, accuracy: {max(scores):.4f}')
print(f'最后5个深度 accuracy: {list(zip(range(16,21), scores[15:20]))}')
plt.figure(figsize=(8, 5))
plt.plot(depths, scores, 'o-', color='#378ADD', lw=2)
plt.axvline(best_d, color='#D85A30', ls='--', label=f'best depth={best_d}')
plt.xlabel('max_depth'); plt.ylabel('CV accuracy')
plt.title('Decision Tree: accuracy vs max_depth')
plt.legend()
plt.tight_layout()
plt.savefig('dt_depth_scan.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
models = {
    'DT(depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'DT(depth=None)': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=5000),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5)
}
names, scores_list = [], []
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5)
    names.append(name)
    scores_list.append(s)
    print(f'{name:22s}: {s.mean():.4f} (+/- {s.std():.4f})')
# 柱状图
plt.figure(figsize=(8, 5))
plt.bar(names, [s.mean() for s in scores_list], color=['r','y','g','b'])
plt.ylabel('CV accuracy')
plt.title('Decision Tree vs Other Classifiers on Digits')
plt.tight_layout()
plt.savefig('dt_vs_models.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()