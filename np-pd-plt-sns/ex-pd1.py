import pandas as pd
import numpy as np

#ex1
s1 = pd.Series([10, 20, 30, 40, 50])
s2 = pd.Series([3.14, 2.71, 1.62], index=['pi', 'e', 'phi'])
s3 = pd.Series({'Beijing': 2154, 'Shanghai': 2487, 'Shenzhen': 1756})

print("默认索引:", s1.values, s1.index)
print("自定义索引:", s2)
print("dict创建:", s3)
# 两种取值方式
print("s2.loc['pi']=", s2.loc['pi'], "| s2.iloc[0]=", s2.iloc[0])
# Series 的 NumPy 操作仍然可用
print("s1 * 2:", s1 * 2)
print("s1 > 25:", s1[s1 > 25])
print("s1.mean():", s1.mean())

#ex2:df的创建
# 方式1：dict of lists
df1 = pd.DataFrame({
    'City': ['北京', '上海', '广州', '深圳'],
    'Population': [2154, 2487, 1868, 1756],
    'Area': [16410, 6340, 7434, 1997]
})
print("dict of lists:\n", df1)
# 方式2：list of dicts
df2 = pd.DataFrame([
    {'Name': 'Alice', 'Age': 25, 'Score': 85},
    {'Name': 'Bob', 'Age': 30, 'Score': 92},
    {'Name': 'Charlie', 'Age': 35, 'Score': 78}
])
print("\nlist of dicts:\n", df2)
# 方式3：从 NumPy 数组
df3 = pd.DataFrame(
    np.random.randn(4, 3),#
    columns=['A', 'B', 'C'],
    index=['row1', 'row2', 'row3', 'row4']
)
print("\nfrom numpy:\n", df3)
# 方式4：自定义 index
df4 = pd.DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
print("\n自定义行索引:\n", df4)

#ex3
df = df1
print("===== head() =====")
print(df.head())#前五行
print("\n===== shape =====")
print(df.shape)
print("\n===== info() =====")
df.info()
print("\n===== describe() =====")
print(df.describe())
print("\n===== dtypes =====")
print(df.dtypes)
print("\n===== columns / index =====")
print("列名:", df.columns.tolist())
print("行索引:", df.index.tolist())
print("\n===== 列即 Series =====")
print(df['Area'])
print(type(df['Area']))
print("单列DataFrame:", type(df[['Area']]))

#ex4
df0 = pd.DataFrame({
    'A': np.random.randn(100),#随机100个标准正态
    'B': np.random.randint(0, 100, 100),#随机0到100
    'C': np.random.choice(['cat', 'dog', 'bird'], 100)#随机分类
})
print(df0)
# NumPy 风格的列运算
df['A_squared'] = df0['A'] ** 2
df['B_normalized'] = (df0['B'] - df0['B'].mean()) / df0['B'].std()

# 条件筛选（布尔索引，和NumPy一样！）
high_a = df0[df0['A'] > 1]
print(f"A>1 的行数: {len(high_a)}")
print(df.describe())#进行count，mean，std，四分位数