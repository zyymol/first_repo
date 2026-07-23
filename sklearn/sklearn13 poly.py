from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
#ex1
X = np.array([[1], [2], [3]])
pf = PolynomialFeatures(degree=3)
X_poly = pf.fit_transform(X)
print(f'原始 shape: {X.shape}  扩展后 shape: {X_poly.shape}')
print(f'列名: {pf.get_feature_names_out()}')
print(X_poly)
#ex2
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.linspace(-5, 5, 150).reshape(-1, 1)#-1自动判断行数多少
y = 0.3*X.ravel()**2 + 2*X.ravel() + 5 + np.random.normal(0, 2, 150)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, d in zip(axes, [1, 2, 5]):
    pipe = Pipeline([('poly', PolynomialFeatures(degree=d)), ('lr', LinearRegression())])
    pipe.fit(X, y)
    r2 = pipe.score(X, y)
    X_plot = np.linspace(-5, 5, 300).reshape(-1, 1)
    y_plot = pipe.predict(X_plot)
    ax.scatter(X, y, s=10, alpha=0.5, c='#0080ff')
    ax.plot(X_plot, y_plot, color='#ff4000', lw=2)
    ax.set_xlim(-6, 6)
    ax.grid(alpha=0.3)
    ax.set_title(f'degree={d}, R^2={r2:.2f}')
fig.suptitle('Poly degree comparison', fontsize=15)
fig.tight_layout()
plt.savefig('13_poly_degree_comparison.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
#ex3
from sklearn.model_selection import cross_val_score
scores=[]
for d in range(1, 9):
    pipe = Pipeline([('poly', PolynomialFeatures(degree=d)), ('lr', LinearRegression())])
    r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()
    scores.append(r2)
    print(f'degree={d}: CV R^2={r2:.4f}')
print(f'BEST: degree {np.argmax(scores)+1}: {scores[np.argmax(scores)]}')