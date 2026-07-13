import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler())
])
X_clean = pipe.fit_transform(df[['temp','humidity','pm25','sales']])
print(f'shape: {X_clean.shape}, mean: {X_clean.mean(axis=0).round(6)}')
#ex2
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler())
    ]), ['temp','humidity','pm25','sales']),
    ('cat', OneHotEncoder(sparse_output=False), ['city'])
])
X = preprocessor.fit_transform(df)
print(f'shape: {X.shape}')
#ex3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X = df[['temp','humidity','pm25','sales','city']]
y = df['promo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe3=Pipeline([
    ('preprocess', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=3))
])
pipe3.fit(X_train, y_train)
print(f'n=3准确率: {pipe3.score(X_test, y_test):.2%}')#50%
#ex4
pipe4=Pipeline([
    ('preprocess', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5))
])
pipe4.fit(X_train, y_train)
print(f'n=5准确率: {pipe4.score(X_test, y_test):.2%}')