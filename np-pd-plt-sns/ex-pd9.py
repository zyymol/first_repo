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
def adjust_sales(row):
    if row['city'] == 'Beijing':
        return row['sales'] * 0.9
    elif row['promo']:
        return row['sales'] * 1.2
    else:
        return row['sales']

df['adjusted'] = df.apply(adjust_sales, axis=1)
print(df[['city','promo','sales','adjusted']].head())