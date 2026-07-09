import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
city=pd.read_csv('C:/Users/24077/ccws/py/city_info.csv',encoding='utf-8-sig' )
df2 = pd.merge(df, city, on='city', how='left')
df2.info()

df2['temp']=df2.groupby('city')['temp'].transform(lambda x: x.fillna(x.median()))
df2['pm25'] = df2.groupby('city')['pm25'].transform(lambda x: x.fillna(x.mean()))
df2.info()

df2['sales_per_capita']=df2['sales']/(df2['population']*10000)
df2['temp_zscore'] = (df2['temp'] - df2['temp'].mean()) / df2['temp'].std()
df2['is_high_sales']=df2['sales']>df2['sales'].median()
df2.info()

df3=df2.groupby('region').agg(sales_sum=('sales','sum'),temp_mean=('temp','mean'),
                              promo_rate=('promo','mean'),pm25_mean=('pm25','mean'))
df3=df3.sort_values('sales_sum',ascending=False)
print(df3.head())
df3.info()
sales_sum_c=df2.groupby('city')['sales'].sum()
print(sales_sum_c)
fig,axes =plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].bar(sales_sum_c.index,sales_sum_c.values)
axes[0,0].set_ylabel('sales sum')
axes[0,0].set_title('fig1: sales sum')
axes[0,0].grid()
axes[0,0].legend()
print(df2['region'].unique())
dregion={'华南':'red','华东':'blue','华北':'black'}
for r in dregion.keys():
    dfsub=df2[df2['region']==r]
    axes[0,1].scatter(dfsub['temp'], dfsub['sales'],
                      color=dregion[r],label=r,alpha=0.5)
axes[0,1].set_title('fig2: temp VS sales')
axes[0,1].set_xlabel('temp')
axes[0,1].set_ylabel('sales')
axes[0,1].grid()
axes[0,1].legend()
sns.boxplot(data=df2, x='city', y='sales',ax=axes[1,0])
axes[1,0].set(xlabel='city')
axes[1,0].set_title('fig3: city sales boxplot')
cor=df2[['temp','sales','pm25','humidity']].corr()
sns.heatmap(cor, annot=True, cmap='coolwarm', vmin=-1, vmax=1,ax=axes[1,1])
axes[1,1].set_title('fig4: correlation heatmap')
fig.suptitle('ANALYSIS')
fig.tight_layout()
fig.savefig('week3pro.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()

print("=== 分析结论 ===")