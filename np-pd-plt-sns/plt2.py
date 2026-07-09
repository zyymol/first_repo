import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1
fig,axes =plt.subplots(2, 2, figsize=(12, 10))
salessum=df.groupby('city')['sales'].sum().sort_values()
axes[0,0].bar(salessum.index,salessum.values)
axes[0,0].set_title('city sales')
axes[0,0].set_ylabel('sales')
axes[0,1].hist(df['sales'], bins=30,alpha=0.3,edgecolor='black')
axes[0,1].set_title('Sales Distribution')
axes[0,1].set_ylabel('sales')
axes[0,1].grid()
axes[1,0].scatter(df['temp'], df['pm25'], alpha=0.5)
axes[1,0].set_xlabel('temp')
axes[1,0].set_ylabel('pm25')
axes[1,0].set_title('Temp VS PM2.5')
promo_counts = df['promo'].value_counts()
axes[1, 1].pie(promo_counts.values, labels=['No Promo','Promo'],autopct='%1.1f%%')
axes[1, 1].set_title('Promo Rate')
fig.suptitle('fig16-1')
fig.tight_layout()
fig.savefig('16-1.png', dpi=150, bbox_inches='tight')
plt.show()

#ex2
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
clst=['Beijing','Guangzhou','Shanghai']
for i in range(3):
    df2=df[df['city']==clst[i]].copy()
    df2=df2.sort_values('date')
    axes[i].plot(df2['date'], df2['temp'], label=clst[i])
    axes[i].set_title(clst[i])
    axes[i].set_xlabel('date')
    axes[i].set_ylabel('temp')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].xaxis.set_major_locator(plt.MaxNLocator(8))
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Temp by City')
fig.tight_layout()
fig.savefig('16-2.png', dpi=150, bbox_inches='tight')
plt.show()

#ex3

df3=df.groupby('date')[['temp','pm25']].mean()
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(df3.index, df3['temp'], color='#378ADD',ls='-',marker='o', linewidth=2, label='Temp')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temp', color='#378ADD')
ax1.tick_params(axis='x', rotation=45)
ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(df3.index, df3['pm25'], color='#E24B4A',ls='--', marker='s',linewidth=2, label='PM2.5')
ax2.set_ylabel('PM2.5',color='#E24B4A')
ax2.legend(loc='upper right')
fig.suptitle('Daily Temp & PM2.5')
fig.tight_layout()
fig.savefig('16-3.png', dpi=150, bbox_inches='tight')
plt.show()

#ex4
df4=df.groupby('city').agg(sales_sum=('sales', 'sum'),temp_mean=('temp', 'mean'),promo_mean=('promo', 'mean'))
fig = plt.figure(figsize=(12, 10))
# 跨越整行
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df3.index, df3['temp'], color='#378ADD',ls='-',marker='o', lw=2, label='Temp')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temp')
ax1.tick_params(axis='x', rotation=45)
ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
ax1.legend()
ax1.set_title('Sub1: Daily temp')

# 第二行左
ax2 = fig.add_subplot(2, 2, 3)
cd={'Beijing':'r','Shanghai':'b','Guangzhou':'y','Shenzhen':'g','Hangzhou':'k'}
for city in df4.index:
    df41=df4[df4.index == city]
    plt.scatter(df41['promo_mean'], df41['sales_sum'],
                c=cd[city], alpha=0.5, label=city)
ax2.set_ylabel('sales_sum')
ax2.set_xlabel('promo_mean')
ax2.legend()
ax2.set_title('Sub2: promo_mean vs sales_sum')

# 第二行右
ax3 = fig.add_subplot(2, 2, 4)
ax3.barh(df4.index,df4['sales_sum'],color=['red','orange','yellow','green','blue'])
ax3.set_xlabel('sales_sum')
ax3.set_title('Sub3: City sales sum')

fig.tight_layout()
fig.savefig('16-4.png', dpi=150, bbox_inches='tight')
plt.show()
