import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 加载鸢尾花数据
iris = load_iris()
x = iris.data
y = iris.target
# 分离训练测试数据
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=43)
# 归一化
std = StandardScaler()
train_x = std.fit_transform(train_x)
test_x = std.fit_transform(test_x)

# svm归一化
clf = SVC()
clf.fit(train_x, train_y)
# 测试准确度
acc = accuracy_score(test_y, clf.predict(test_x))
print(acc)
# 保存模型
joblib.dump(clf, './model/iris_svm.pkl')
# # 使用模型
# model = joblib.load('./model/iris_svm.pkl')
