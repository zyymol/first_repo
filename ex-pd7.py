import pandas as pd
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
df.info()
print(df.describe())
#缺失看Non-Null Count，范围看min和max
#ex2
df['temp'] = df.groupby('city')['temp'].transform(lambda x: x.fillna(x.median()))
df['pm25']=df['pm25'].fillna(df['pm25'].mean())#inplace默认False只是副本需要赋值回去，True是直接在原表上改不赋值
df=df.dropna(subset=['sales'])
df.info()

#ex3
df['sales_zscore'] = (df['sales'] - df['sales'].mean()) / df['sales'].std()
df['pm25_z']=(df['pm25'] - df['pm25'].mean()) / df['pm25'].std()
df['composite_score'] = 0.4 * df['sales_zscore'] - 0.3 * df['pm25_z'] + 0.3 * df['promo']
print(df.head())

#ex4
dfres=df.groupby('city').agg(sales_sum=('sales', 'sum'),
    sales_mean=('sales', 'mean'),
    promo_mean=('promo', 'mean'),
    composite_score_mean=('composite_score', 'mean'))
print(dfres.sort_values('composite_score_mean', ascending=False))