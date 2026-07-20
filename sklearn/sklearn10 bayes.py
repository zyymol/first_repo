from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
#ex1
X, y = load_digits(return_X_y=True)
gnb = GaussianNB()#model
scores = cross_val_score(gnb, X, y, cv=5)
print(f'GaussianNB CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')
gnb.fit(X, y)
print(f'先验概率: {gnb.class_prior_.round(3).tolist()}')
#ex2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'LogisticReg': LogisticRegression(max_iter=5000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVC(rbf)': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
names, means = [], []
for name, m in models.items():
    # 注意：NB 不需要 scaler，其他模型需要决策——但 CV 内 NB 直接用原始数据
    # 为公平对比，每个模型都用原始数据（不做 scaler），只测 baseline
    s = cross_val_score(m, X, y, cv=5)
    names.append(name)
    means.append(s.mean())
order = np.argsort(means)[::-1]
names_sorted = [names[i] for i in order]
means_sorted = [means[i] for i in order]

plt.figure(figsize=(10, 5))
bars = plt.bar(names_sorted,means_sorted,color=['#7F77DD','#D85A30','#639922','#378ADD','#888780','#F0997B'])
for bar, val in zip(bars, means_sorted):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', fontsize=11)
plt.ylabel('accuracy')
plt.tight_layout()
plt.savefig('nb_vs_all.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
gnb = GaussianNB().fit(X, y)
print(f'theta_ shape: {gnb.theta_.shape}')#(10, 64)：10个数字 × 64个像素
print(f'var_ shape:   {gnb.var_.shape}')
print()
print('数字 0 的前5个像素:')
print(f'  均值 = {gnb.theta_[0, :5].round(2)}')
print(f'  方差 = {gnb.var_[0, :5].round(2)}')