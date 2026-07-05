import numpy as np
#ex1
arr = np.arange(36).reshape(6, 6)
print("6x6 数组:\n", arr)
# 基本索引
print("\n中心2x2:", arr[2:4, 2:4])
print("左上3x3:", arr[:3, :3])
print("右下2x3:", arr[-2:, -3:])
# 步长切片
print("\n隔行隔列(步长2):\n", arr[::2, ::2])
print("反向:\n", arr[::-1, ::-1])
# 特定行列
print("\n第3行:", arr[2])
print("第4列:", arr[:, 3])
print("第2-4行，第1-3列:\n", arr[1:4, :3])

# ex2：切片是视图，指向同一个地址
a = np.array([1, 2, 3, 4, 5],dtype=np.int32)
view = a[1:4]
view[0] = 999
print("原数组被改了:", a)#[1 999 3 4 5]
print("view是视图?", view.base is a)#True

# 布尔索引：条件选取出个copy，赋值后不在同一地址
a1 = np.array([1, 2, 3, 4, 5],dtype=np.int32)
copy = a1[a1 > 2]
copy[0] = 999
print("原数组没变:", a1) # [1 2 3 4 5]
print(copy)#[999   4   5]
print("布尔索引是副本?", np.shares_memory(a1, copy))#False, 是副本
a1[a1 > 2]=0
print(a1)#改的是原数组
a1[a1==0]=[3,4,5]
print(a1)
# 花式索引也是副本
fancy = a1[[0, 2, 4]]
fancy[0] = 999
print("原数组没变:", a1)

print("Summary:切片a[:]是视图；布尔索引a[>]是副本；花式索引a[[]]是副本\n")

#ex3
np.random.seed(42)
scores = np.random.randint(40, 100, 20)  # 20个随机成绩
print("全部成绩:", scores)

# 不及格的
print("不及格:", scores[scores < 60])

# 优秀（>=90）
print("优秀:", scores[scores >= 90])

# 中等（60-79）
print("中等:", scores[(scores >= 60) & (scores < 80)])

# 统计
print(f"平均分: {scores.mean():.1f}")
print(f"最高: {scores.max()}, 最低: {scores.min()}")
print(f"不及格人数: {len(scores[scores < 60])}")
print(f"优秀人数: {len(scores[scores >= 90])}")

# 用布尔索引修改元素（注意：这是布尔索引赋值，修改的是原数组）
scores[scores < 60] = 0  # 不及格的清零
print("\n处理后:", scores)

# 组合条件 + 花式索引 综合练习
a = np.random.randint(0, 20, (5, 5))
print("\n5x5 随机矩阵:\n", a)
# 找出所有 >10 的元素的位置
mask = a > 10
rows, cols = np.where(mask)#mask每个元素在a的位置
print(">10 的元素位置 (row, col):")
for r, c in zip(rows, cols):
    print(f"  [{r},{c}] = {a[r, c]}")

#ex4
# 练习：用花式索引高效地打乱一个数组的行顺序
a = np.arange(20).reshape(4, 5)
print("原始:\n", a)
print(a.T)

# 用花式索引重新排序
new_order = [3, 0, 2, 1]  # 自定义行顺序
shuffled = a[new_order]#shuffled[0]这一行就是a[3]
print("\n按行重排后:\n", shuffled)

# 取对角线元素（花式索引 + arange）
diag_idx = np.arange(4)
print("\n对角线:", a[diag_idx, diag_idx])

# 更复杂：取 (0,1), (2,3), (1,4) 三个位置
rows = [0, 2, 1]
cols = [1, 3, 4]
print("指定位置:", a[rows, cols])

# 挑战：用花式索引实现矩阵的行列交换
temp = a.copy()
temp[[0, 3]] = temp[[3, 0]]# 把第0行和第3行交换
print("\n交换第0行和第3行:\n", temp)