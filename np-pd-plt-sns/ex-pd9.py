import pandas as pd
#ex1
df=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv',
               parse_dates=['date'],encoding='utf-8-sig')
df.info()
df['sales_grade']=df['sales'].apply(lambda x: 'low' if x < 15000 else ('high' if x > 35000 else 'mid'))

city_region = {'Beijing':'华北','Shanghai':'华东','Guangzhou':'华南','Shenzhen':'华南','Hangzhou':'华东'}
df['region'] = df['city'].map(city_region)
print(df[['sales','sales_grade','region']].head())

#ex2
df['sales_ma7'] = df['sales'].rolling(7).mean()
print(df[['sales','sales_ma7']].tail(10))

#ex3
df['sales_z']=(df['sales']-df['sales'].mean())/df['sales'].std()
df['pm25_z']=(df['pm25']-df['pm25'].mean())/df['pm25'].std()
df['score'] = df.apply(lambda row: row['sales_z'] + row['promo'] * 0.3 - row['pm25_z'], axis=1)
print(df.head())