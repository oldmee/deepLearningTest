import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

X = np.linspace(-2, 6, 200) # 创建200个 -2 到 6 的等距向量
np.random.shuffle(X) # shuffle将X随机排列
Y = 0.5 * X + 2 + 0.15 * np.random.randn(200, )  # 创建一组由 y = 0.5x + 2 加上一些噪声而生成的数据

# plot data
plt.scatter(X, Y, 50, 'y') # 这是matlab画散点图的命令；这里X，Y是画散点图的数据。 其中S为大小，系统默认大小为50，C表示所画图的颜色
plt.show()

X_train, Y_train = X[:160], Y[:160]  # train first 160 data points
X_test, Y_test = X[160:], Y[160:]  # test remaining 40 data points


model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))
model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X_train, Y_train, epochs=100, batch_size=64)

cost = model.evaluate(X_test, Y_test, batch_size=40)

print('\nTesting ------------')
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

