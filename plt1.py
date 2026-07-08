import pandas as pd
import matplotlib.pyplot as plt
#ex1
df=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig', parse_dates=['date'])
df1=df.sort_values('date').iloc[:30]

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
ax1.plot(df1['date'],df1['sales'],'o-',label='sales',color='r',lw=2)
ax1.set_ylabel('sales')
ax1.set_xlabel('date')
ax1.legend()
ax2.plot(df1['date'],df1['temp'],'o-',label='temp',color='b',lw=2)
ax2.set_ylabel('temp')
ax2.set_xlabel('date')
ax2.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout()
plt.savefig('temp_sales.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()

#ex2
cd={'Beijing':'r','Shanghai':'b','Guangzhou':'y','Shenzhen':'g','Hangzhou':'k'}
plt.figure(figsize=(8, 6))
for city in df['city'].unique():#unique去重
    df2=df[df['city'] == city]
    plt.scatter(df2['temp'], df2['pm25'],
                c=cd[city], alpha=0.5, label=city, s=40)
plt.xlabel('temp')
plt.ylabel('pm25')
plt.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout()
plt.savefig('temp_pm25_scatter.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()

#ex3
df3=df.groupby('city')['sales'].sum().sort_values(ascending=False)
plt.barh(df3.index, df3.values,#h是横向的意思
         color=['red','orange','yellow','green','blue'])
plt.xlabel('Sales Sum')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('city_sales_bar.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()

#ex4
dmean = df.groupby('date')[['temp','pm25']].mean()
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(dmean.index,dmean['temp'], color='b', linewidth=2, label='temp')
ax1.set_xlabel('date')
ax1.set_ylabel('temp')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(dmean.index,dmean['pm25'], color='r', linewidth=2, label='pm25')
ax2.set_ylabel('pm25')
ax2.legend()
plt.tight_layout()
plt.savefig('temp_mean & pm25_mean.png', dpi=150, bbox_inches='tight', facecolor='none')
plt.show()
