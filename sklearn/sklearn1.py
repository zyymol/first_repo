from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
sub=df[['temp','humidity','sales']]
im=SimpleImputer(strategy='median')
X_clean = im.fit_transform(sub)
print('before\n',sub.mean().values)
print(sub.std().values)
scaler = StandardScaler()
sub_scaled = scaler.fit_transform(sub)
df1=pd.DataFrame(sub_scaled,columns=['temp','humidity','sales'])
print('after\n',df1.mean().values)
print(df1.std().values)
#ex2
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # 返回普通数组（非稀疏矩阵）
city_arr=encoder.fit_transform(df[['city']])
df2=pd.DataFrame(city_arr,columns=encoder.categories_[0])
print(df2.shape)
print(df2[:5])
#ex3
imp3= SimpleImputer(strategy='median')
temp_imputed = imp3.fit_transform(df[['temp']])
print(f'before: {df["temp"].isnull().sum()}')
print(f'after: {pd.DataFrame(temp_imputed).isnull().sum().sum()}')
#ex4

imp4=SimpleImputer(strategy='median')
df_imputed =imp4.fit_transform(df[['temp','humidity','pm25','sales']])
df43=StandardScaler().fit_transform(df_imputed)
df42=pd.DataFrame(df43,columns=['temp','humidity','pm25','sales'])
df4=pd.concat([df2,df42],axis=1)
print(df4.shape, df4.isnull().sum().sum())
print(df4.head())