import pandas as pd
import numpy as np
#ex1
df1 = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
print(df1.head())
df1.info()
print(df1.describe())

#ex2
df2 = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', nrows=50,usecols=['city', 'temp', 'sales'])
print(df2.head())

#ex3
df3=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv',
                parse_dates=['date'],index_col='date' )#不设置index_col则默认是RangeIndex
print(df3.dtypes)
print(df3.index)
df33=df3.loc['2026-03-01':'2026-03-31']
print(df33.head())

#ex4
df4=df1[(df1['temp']>25)&(df1['sales']>20000)]
print(df4.head())
df4.dropna()
df4.to_csv('hot_sales.csv',encoding='utf-8-sig',index=False)
print('success,len=',len(df4))
