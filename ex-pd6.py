import pandas as pd
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
df1=df[(df['temp']>15)&(df['temp']<25)&(df['city'].isin(['Shanghai','Beijing']))&(df['sales']>30000)]
print(df1)

#ex2
df21=df.sort_values(['city', 'sales'], ascending=[True, False])
print(df21)
df22=df.nlargest(5, 'sales')
print(df22)

#ex3
df['sales_rank'] = df['sales'].rank(ascending=False)
print(df.sort_values('sales_rank', ascending=True))

#ex4
df4=df.query('temp > 20 and humidity < 50 and promo==True')
print(df4.sort_values('sales', ascending=False))
print(f'共{len(df4)}条')