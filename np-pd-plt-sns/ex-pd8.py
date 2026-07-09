import pandas as pd
#ex1
s1 = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
s2 = pd.read_csv('C:/Users/24077/ccws/py/new_sales.csv', encoding='utf-8-sig')
df1 = pd.concat([s1, s2], ignore_index=True)
print("index change:%d %d"%(len(s1),len(df1)))
df1.info()

#ex2
city=pd.read_csv('C:/Users/24077/ccws/py/city_info.csv',encoding='utf-8-sig' )
df2 = pd.merge(s1, city, on='city', how='left')
df2.info()

#ex3
for how in ['inner','left','right','outer']:
    pd3 = pd.merge(s1,s2,on=['city','date'], how=how,
                 suffixes=('_old','_new'))
    print(f"{how:6s}: {len(pd3)} row")#6s保证对齐6个格
    print(pd3.head(3))
#inner: 匹配成功的，为0；left:  200（sales全保留，new匹配不上NaN）；right: 10（new全保留）；outer: 210（两表全部）

#ex4
df41=s1.set_index('city')
df42=city.set_index('city')
r_out=df41.join(df42,how='outer')
print('outer join')
r_out.info()

r_in=df41.join(df42,how='inner')
print('inner join')
r_in.info()