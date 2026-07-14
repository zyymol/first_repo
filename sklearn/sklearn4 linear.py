from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error, r2_score
#ex1
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R²:  {r2_score(y_test, y_pred):.3f}')
#ex2
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='r2')
print(f'CV scores: {scores}')
print(f'Mean R²: {scores.mean():.3f}')
print(f'Std:     {scores.std():.3f}')
from sklearn.pipeline import Pipeline
#ex3
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f'Pipeline CV R²: {scores.mean():.3f} (+/- {scores.std():.3f})')
#ex4
pipe4=Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1))
])
scores4=cross_val_score(pipe4, X, y, cv=5, scoring='r2')
print(f'Pipeline CV R²: {scores4.mean():.3f} (+/- {scores4.std():.3f})')