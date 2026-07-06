import pandas as pd
import numpy as np
#ex1
df = pd.DataFrame(
    {'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
     'Age': [25, 30, 35, 28, 22],
     'Score': [85, 92, 78, 88, 95]},
    index=['one', 'two', 'three', 'four', 'five']
)
print("原始:\n", df)
# 关键对比：loc vs iloc 切片区间
print("\nloc['two':'four'] (闭区间, 3行):")
print(df.loc['two':'four'])
print("\niloc[1:4] (半开区间, 3行):")
print(df.iloc[1:4])
# 行列组合选择
print("\nloc[['one','three'], ['Name','Score']]:")
print(df.loc[['one', 'three'], ['Name', 'Score']])
# 数字索引陷阱演示
df2 = df.reset_index(drop=True)  # index 变成 0,1,2,3,4
print("\n--- 数字index时的loc vs iloc差异 ---")
print(df2)
print(f"loc[1:3] 返回: {df2.loc[1:3].index.tolist()}  (3行，闭区间)")
print(f"iloc[1:3] 返回: {df2.iloc[1:3].index.tolist()}  (2行，半开区间)")#1，2

#ex2
np.random.seed(42)
df = pd.DataFrame({
    'Name': ['A' + str(i) for i in range(20)],
    'Age': np.random.randint(18, 60, 20),
    'Score': np.random.randint(40, 100, 20),
    'City': np.random.choice(['北京', '上海', '广州', '深圳'], 20)
})

# 单条件：Age>30
print("Age > 30:\n", df[df['Age'] > 30].head())
print(f"共 {len(df[df['Age'] > 30])} 人")
# 多条件:用&
print("\nAge > 30 且 Score > 70:")
print(df[(df['Age'] > 30) & (df['Score'] > 70)])
# isin 集合筛选
print("\n只看北京和上海:")
print(df[df['City'].isin(['北京', '上海'])])
# ~ 取反
print("\n不是北京的:")
print(df[~df['City'].isin(['北京'])])
# loc + 布尔修改（安全写法）：Score<60的都变成60
df.loc[df['Score'] < 60, 'Score'] = 60
print("\n修改后（Score<60的都变成60）:")
print(df[df['Score'] == 60])
# query 方法：代替多条件
print("\nquery版本:")
print(df.query('Age > 30 and Score > 70'))

#ex3
import warnings
warnings.simplefilter('error')  # 让警告变成错误，看得更清楚
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
try:
    df[df['A'] > 1]['B'] = 0  # 链式索引是按要求搞出副本，原来的df不动，触发warning
except Exception as e:
    print(f"错误: {type(e).__name__}")
# 正确写法
df.loc[df['A'] > 1, 'B'] = 0   # loc一步完成，把A列大于1的行的B列变成0
print("\n正确写法结果:\n", df)

#ex4：模拟一个学生成绩表，完成综合查询
np.random.seed(42)
df = pd.DataFrame({
    'name': [f'student_{i}' for i in range(100)],
    'math': np.random.randint(40, 100, 100),
    'english': np.random.randint(40, 100, 100),
    'class': np.random.choice(['A', 'B', 'C'], 100)
})
df['avg'] = (df['math'] + df['english']) / 2
# 查询1：A班数学成绩前5
print("A班数学前5:")
print(df[df['class'] == 'A'].nlargest(5, 'math')[['name','class','math']])
# 查询2：平均分>80的人数（按班级）
for i in ['A', 'B', 'C']:
    sub = df[(df['class'] == i) & (df['avg'] > 80)]
    print(f"{i}班平均>80: {len(sub)}人")
# 查询3：用 loc 把所有C班同学的数学分+5
df.loc[df['class'] == 'C', 'math'] += 5
df.loc[df['class'] == 'C', 'avg'] = (df['math'] + df['english']) / 2
print("\nC班加分后:", df[df['class']=='C'][['name','math','english','avg']].head())
