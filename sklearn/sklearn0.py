from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. 创建模型
model = KNeighborsClassifier(n_neighbors=3)

# 4. 训练（fit）
model.fit(X_train, y_train)

# 5. 预测（predict）
y_pred = model.predict(X_test)

# 6. 评估（score）
accuracy = model.score(X_test, y_test)
print(f"测试集准确率: {accuracy:.2%}")
print(f"预测: {y_pred[:10]}")
print(f"真实: {y_test[:10]}")