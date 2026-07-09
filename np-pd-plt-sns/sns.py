import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/24077/ccws/py/sales_data.csv', encoding='utf-8-sig')
#ex1: boxplot看分组分布&异常
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='city', y='sales')
plt.title('City Sales Distribution')
plt.tight_layout()
plt.savefig('sns1.png', dpi=150, bbox_inches='tight')
plt.show()

#ex2: violinplot看分布形状
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='city', y='temp', inner='quartile')
plt.title('CIty Temp Distribution')
plt.tight_layout()
plt.savefig('sns2.png', dpi=150, bbox_inches='tight')
plt.show()

#ex3: heatmap看（相关性）矩阵模式
cor=df[['temp','humidity','pm25','sales']].corr()
sns.heatmap(cor,annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig( 'sns3.png', dpi=150, bbox_inches='tight')
plt.show()

#ex4:
sns.pairplot(df[['temp','humidity','pm25','sales']], height=2)
plt.suptitle('Pairwise Relationships',y=1.002)
plt.savefig('sns4_pairplot_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

sns.boxplot(data=df, x='city', y='sales', hue='promo')#hue作为二级分类
plt.title('Sales by City and Promo Status')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('sns4_boxplot_promo.png', dpi=150, bbox_inches='tight')
plt.show()