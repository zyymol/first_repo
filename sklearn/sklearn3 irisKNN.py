from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
#ex1
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=3))
])
pipe.fit(X_train, y_train)
print( f'acc: {pipe.score(X_test, y_test):.3%}')
#ex2
from sklearn.metrics import confusion_matrix, classification_report
y_pred = pipe.predict(X_test)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print()
print('Classification Report:\n', classification_report(y_test, y_pred,
      target_names=iris.target_names))
#ex3
scores = []
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    print(f'k={k:2d}: {scores[-1]:.2%}')
print(f'\nBest k={scores.index(max(scores))+1}, score={max(scores):.2%}')