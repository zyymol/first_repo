import numpy as np

#ex1
a1d = np.array([10, 20, 30, 40, 50])
a2d = np.array([[1, 2, 3], [4, 5, 6]])
a3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("1D:", a1d.shape, "| 2D:", a2d.shape, "| 3D:", a3d.shape)


z = np.zeros((3, 4))
o0=np.zeros_like(a2d)
o1 = np.ones_like(a2d)#像a2d形状的全是1
print("zeros(3,4):", z.shape, "\n| ones_like:", o1)
print(o0)


print("arange:", np.arange(5, 15, 2))

print("arange 0-1:", np.arange(0, 1, 0.2))
print("linspace 0-1:", np.linspace(0, 1, 5))

np.random.seed(42)
r1 = np.random.random((2, 3))
r2 = np.random.randint(0, 100, 5)
print("uniform:", r1)
print("randint:", r2)


print("eye:", np.eye(3))
print("diag:", np.diag([5, 6, 7]))

#ex2
arrays = [
    np.array([1, 2, 3, 4, 5]),
    np.array([[1, 2], [3, 4], [5, 6]]),
    np.arange(24).reshape(2, 3, 4),      # 三维！
    np.ones((5, 5), dtype=np.float32),   # 指定dtype
    np.zeros((100, 100, 3), dtype=np.uint8)
]

for i, arr in enumerate(arrays):
    print(f"Array {i+1}:")
    print(f"  shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}")
    print(f"  size={arr.size}, itemsize={arr.itemsize}B, nbytes={arr.nbytes}B")
    print()

arr12 = np.arange(12)
print("原始:", arr12, arr12.shape)
print("2x6:", arr12.reshape(2, 6))
print("3x4:", arr12.reshape(3, 4))
print("2x2x3:", arr12.reshape(2, 2, 3))

#ex3
print("int default:", np.array([1, 2, 3]).dtype)#int64
print("float default:", np.array([1.0, 2.0]).dtype)#float64
print("mixed ->:", np.array([1, 2.0, 3]).dtype)#float64
print("as float32:", np.array([1, 2, 3], dtype=np.float32))
print("as uint8:", np.array([1, 2, 3], dtype=np.uint8))
a = np.array([1.2, 3.7, 5.9])
print("原始:", a, a.dtype)
print("->int:", a.astype(int), a.astype(int).dtype)  # 截断！成了int默认的int64
print("->int32:", a.astype(np.int32))

a64 = np.ones(1000000, dtype=np.float64)
a32 = np.ones(1000000, dtype=np.float32)
a8  = np.ones(1000000, dtype=np.uint8)
print(f"float64: {a64.nbytes / 1e6:.1f} MB")#8M
print(f"float32: {a32.nbytes / 1e6:.1f} MB")#4M
print(f"uint8:   {a8.nbytes / 1e6:.1f} MB")#1M

#ex4
import time

n = 10_000_000
a_list = list(range(n))
b_list = list(range(n))

# Python list 加法（慢）
start = time.time()
c_list = [a_list[i] + b_list[i] for i in range(n)]
print(f"list for-loop: {time.time() - start:.3f}s")#0.778s

# ndarray 向量化加法（快）
a_arr = np.array(a_list)
b_arr = np.array(b_list)
start = time.time()
c_arr = a_arr + b_arr
print(f"ndarray vec:    {time.time() - start:.3f}s")#0.019s