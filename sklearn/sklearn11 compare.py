from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

models = {
    'GaussianNB':(GaussianNB(), X),
    'DT':(DecisionTreeClassifier(random_state=42), X),
    'LR':(LogisticRegression(max_iter=5000), Xs),
    'RF':(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X),
    'SVC(rbf)':(SVC(kernel='rbf', random_state=42), Xs),
    'KNN':(KNeighborsClassifier(n_neighbors=5), Xs),
}
res=[]
for name, (model, data) in models.items():
    acc = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    f1  = cross_val_score(model, data, y, cv=5, scoring='f1')
    res.append({
        'Model': name,
        'Accuracy': acc.mean(),
        'Acc_std': acc.std(),
        'F1': f1.mean(),
        'F1_std': f1.std()
    })
import pandas as pd
df = pd.DataFrame(res).sort_values('F1', ascending=False)
print(df.round(4))
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'].tolist(),df['F1'].tolist(),yerr=df['F1_std'].tolist(),capsize=5,color=['red','orange','yellow','green','blue','grey'])
for bar, val in zip(bars, df['F1'].tolist()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
             f'{val:.3f}', ha='center', fontsize=10)
plt.ylabel('F1')
plt.tight_layout()
plt.savefig('all_model_F1.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()