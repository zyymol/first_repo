import pandas as pd
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
df1=df.groupby('city')['sales'].agg(['sum','mean','std'])
print(df1.sort_values('sum', ascending=False))

#ex2
df2=df.groupby('city').agg({'sales': 'sum','temp': 'mean','humidity': 'median','promo': 'mean'})
print(df2.sort_values('sales', ascending=False))

#ex3
print(df.sort_values('sales', ascending=False).head(10))
print(df.sort_values('sales', ascending=False).head(10)['date'])
df3=df.groupby('city')['sales'].agg(['sum'])
print(df3.sort_values('sum', ascending=False))#总量
print(df.groupby('city')['sales'].mean().sort_values(ascending=False))#?平均销售额

#ex4
df4=df.copy()
df4['city_sales_median'] = df4.groupby('city')['sales'].transform('median')
df4s=df4[df4['sales']>df4['city_sales_median']]
print(df4s.groupby('city')['sales'].agg(['count']))