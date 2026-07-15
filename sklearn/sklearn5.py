from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te  = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_tr, y_train)
y_pred = model.predict(X_te)
cm = confusion_matrix(y_test, y_pred)# 混淆矩阵
print("Confusion Matrix:",cm)
print()
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]
#print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
precision_manual = TP / (TP + FP)
recall_manual    = TP / (TP + FN)
f1_manual        = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)
print(f"手动 precision = {precision_manual:.4f}")
print(f"手动 recall    = {recall_manual:.4f}")
print("classification_report（验证）:")
print(classification_report(y_test, y_pred, target_names=['benign', 'malignant']))
#ex2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
y_prob = model.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, color='r', label=f'KNN(k=5) AUC={auc:.3f}')
plt.plot([0, 1], [0, 1], '--', lw=2, color='grey',label='Random guess AUC=0.5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Breast Cancer')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score

models = {
    'KNN(k=1)': KNeighborsClassifier(n_neighbors=1),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN(k=15)': KNeighborsClassifier(n_neighbors=15),
    'LogisticRegression': LogisticRegression(max_iter=5000)
}
results = []
for name, model in models.items():
    # accuracy（CV）
    acc = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy').mean()
    # F1（CV）
    y_cv_pred = cross_val_predict(model, X_tr, y_train, cv=5)
    f1 = f1_score(y_train, y_cv_pred)
    # AUC（CV）
    y_cv_prob = cross_val_predict(model, X_tr, y_train, cv=5, method='predict_proba')[:, 1]
    auc = roc_auc_score(y_train, y_cv_prob)
    results.append({'model': name, 'accuracy': acc, 'F1': f1, 'AUC': auc})
df = pd.DataFrame(results)
print(df.sort_values('AUC', ascending=False))

