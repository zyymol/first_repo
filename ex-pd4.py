import pandas as pd
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
df.info()#Non-null数量/200entries
print(df.isnull().sum())
print(df.isnull().sum()/len(df))

#ex2
print('Origin:',len(df))
print('dropna():',len(df.dropna()))
print('dropna(thresh=6):',len(df.dropna(thresh=6)))#一行6个值及以上非空就保留
print('only drop temp na:',len(df.dropna(subset=['temp'])))
print(len(df))

#ex3
tempf=df['temp'].copy()
f_mean=tempf.fillna(tempf.mean())
print(f_mean.describe())
f_median = tempf.fillna(tempf.median())
print(f_median.describe())
f_ffill  = tempf.fillna(method='ffill')
print(f_ffill.describe())
f_bfill  = tempf.fillna(method='bfill')
print(f_bfill.describe())

#ex4
df4=df.copy()
print('ex4\n')
print(df4.describe())
df4['sales'] = df4.groupby('city')['sales'].transform(lambda x: x.fillna(x.median()))
print(df4.describe())