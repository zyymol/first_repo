import numpy as np
import time

np.random.seed(42)
start1 = time.time()
A = np.random.randn(3, 3) + np.eye(3) * 2   # 确保可逆
A_inv = np.linalg.inv(A)
print("A:\n", A)
print("A_inv:\n", A_inv)
print("A @ A_inv = I?\n", (A @ A_inv).round(8))#单位阵，@是作用
print("det(A) =", np.linalg.det(A))

eigvals, eigvecs = np.linalg.eig(A)
print("特征值:", eigvals)
print("验证第一个特征值: |A v - lambda v| =",
      np.linalg.norm(A @ eigvecs[:, 0] - eigvals[0] * eigvecs[:, 0]).round(10))#得到向量模长为0
print(f'ex1 time: {time.time() - start1:.4f}s')#0.0010s

#ex2
np.random.seed(42)
start2 = time.time()
n = 100
X_raw = np.random.randn(n, 2)#(n,2)型，每个元素满足标准正态
y = 3 * X_raw[:, 0] + 2 * X_raw[:, 1] + 1 + np.random.randn(n) * 0.5
#数据: y = 3*x1 + 2*x2 + 1 + noise
# 加截距列
X = np.column_stack([np.ones(n), X_raw])

# 正规方程: beta = (X^T X)^(-1) X^T y
beta = np.linalg.solve(X.T @ X, X.T @ y)
print(f"系数: beta0(截距)={beta[0]:.3f}, beta1={beta[1]:.3f}, beta2={beta[2]:.3f}")
print(f"期望: 1.0, 3.0, 2.0")

y_pred = X @ beta
mse = np.mean((y - y_pred) ** 2)#残差MSE
print(f"MSE: {mse:.4f}  (噪声方差约 0.25)")
print(f'ex2 time: {time.time() - start2:.4f}s')#0.0002s

#ex3
# 模拟：用 SVD 低秩近似一个矩阵
start3 = time.time()
A = np.random.randn(10, 8)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("原始形状:", A.shape, "奇异值:", S.round(2))
print("奇异值衰减:", (S / S.max()).round(3))

# 用前k个奇异值重构
for k in [1, 3, 5, 7, 8]:#k=8压缩比0.5已降到0.0000
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k) / np.linalg.norm(A)
    print(f"k={k}: 相对误差={error:.4f}, 压缩比={A.size/((10+8+1)*k):.1f}x")
print(f'ex3 time: {time.time() - start3:.4f}s')#0.0004s