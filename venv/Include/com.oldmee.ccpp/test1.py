import pandas as pd
import numpy as np
data = pd.read_csv(r'ccpp.csv', header=0)

data.head()
data.shape

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]

# 首先从sklearn库中导入划分函数
from sklearn.model_selection import train_test_split
# 然后执行函数获得结果
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

# 首先从sklearn库中导入线性回归函数
from sklearn.linear_model import LinearRegression
# 执行函数获得一个线性回归模型
linreg = LinearRegression()  # 这是一个未经训练的机器学习模型
# 对模型传入输入数据x_train和输出数据y_train
linreg.fit(x_train, y_train)  # 这是一个经过训练的机器学习模型
'''输出线性回归的截距和各个系数'''
print('linreg.intercept_: ', linreg.intercept_)
print('linreg.coef_: ', linreg.coef_)

y_pred = linreg.predict(x_test)
# 引入sklearn模型评价工具库
from sklearn import metrics
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# 这是为了能在交互式界面中显示图像
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()