import numpy as np
#ex1
a = np.arange(24)
# reshape 组合
print("(2,12):", a.reshape(2, 12))
print("(2,3,4):", a.reshape(2, 3, 4))
print("-1用法:", a.reshape(3, -1))   # 自动推导→(3,8)

# flatten vs ravel
b = np.array([[1, 2], [3, 4]])
f = b.flatten()
r = b.ravel()
f[0] = 999
print("flatten影响原数组?False\n", b) # 没变
print(f)
r[0] = 999
print("ravel影响原数组?True\n", b) # 变了
print(r)

# transpose转置
c = np.arange(6).reshape(2, 3)
print("转置前:\n", c)
print("转置后:\n", c.T)

x = np.array([1, 2, 3, 4])
print("行向量:", x[np.newaxis, :], x[np.newaxis, :].shape)#(1,4)
print("列向量:", x[:, np.newaxis], x[:, np.newaxis].shape)#(4,1)

# squeeze
d = np.array([[[1, 2, 3]]])
print("squeeze:", d.squeeze().shape)  # (3,)压到最低维

#ex2 拼接
a2 = np.array([[1, 2], [3, 4]])
b2 = np.array([[5, 6], [7, 8]])

print("垂直拼接:\n", np.concatenate([a2, b2]))#默认axis=0
print("水平拼接:\n", np.concatenate([a2, b2], axis=1))

print("vstack:\n", np.vstack([a2, b2]))
print("hstack:\n", np.hstack([a2, b2]))

# stack 与 concatenate 的区别
print("stack (新轴):", np.stack([a2, b2]).shape)       # (2, 2, 2)
print("concatenate:", np.concatenate([a2, b2]).shape)  # (4, 2)

# 划分数组（反向操作）
big = np.arange(12).reshape(4, 3)
top, bottom = np.vsplit(big, 2)
print("vsplit: top\n", top, "\nbottom\n", bottom)

#ex3
# 实验1：预判训练——不用电脑，先在纸上写出结果 shape
A1 = np.ones((5, 3)); B1 = np.ones((3,))
print("(5,3)+(3,):", (A1 + B1).shape)   # (5,3) ✓

A2 = np.ones((5, 3)); B2 = np.ones((5, 1))
print("(5,3)+(5,1):", (A2 + B2).shape)  # (5,3) ✓

A3 = np.ones((5, 3)); B3 = np.ones((5,))
try: x=(A3 + B3).shape
except ValueError as e: print("(5,3)+(5,): ValueError")#出e

A4 = np.ones((2, 3, 4)); B4 = np.ones((3, 4))
print("(2,3,4)+(3,4):", (A4 + B4).shape)

# 实验2：中心化（广播最经典应用）
data = np.random.randn(1000, 5)
mean = data.mean(axis=0)       # (5,)
centered = data - mean         # (1000,5) - (5,) → 广播！
print("中心化验证:", centered.mean(axis=0).round(5))#接近均值，结果为0

# 实验3：归一化
std = data.std(axis=0)         # (5,)
normalized = centered / std    # (1000,5) / (5,) → 广播！
print("归一化std:", normalized.std(axis=0).round(3))

# 实验4：外积
a = np.array([1, 2, 3])
b = np.array([10, 20])
outer = a[:, np.newaxis] * b
print("外积 (3,2):\n", outer)

# 实验5：网格数据（meshgrid替代）
x = np.linspace(0, 1, 4)      # (4,)
y = np.linspace(0, 1, 3)      # (3,)
grid = x + y[:, np.newaxis]   # (4,) + (3,1) → (3,4)
print("网格:\n", grid)

#ex4
import time
X = np.random.randn(500, 3)   # 500个3D点
Y = np.random.randn(300, 3)   # 300个3D点

start1 = time.time()
dist = np.zeros([500, 300])
for i in range(500):
    for j in range(300):
        dist[i, j] = np.sqrt(((X[i] - Y[j]) ** 2).sum())
print("距离矩阵 shape:", dist.shape)                # (500,300)
print("验证 (0,0) =", np.sqrt(((X[0] - Y[0]) ** 2).sum()), end="")
print(" =?", dist[0, 0])
print(f"list for-loop: {time.time() - start1:.4f}s")#0.4660s
start2=time.time()
diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (500,1,3) - (1,300,3) → (500,300,3)
dist2 = np.sqrt((diff ** 2).sum(axis=2))             # (500,300)
print("距离矩阵 shape:", dist2.shape)                # (500,300)
print("验证 (0,0) =", np.sqrt(((X[0] - Y[0]) ** 2).sum()), end="")
print(" =?", dist2[0, 0])
print(f"ndarray: {time.time() - start2:.4f}s")#0.0041s