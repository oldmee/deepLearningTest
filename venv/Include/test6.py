import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2, 6, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + 0.15 * np.random.randn(200, )

# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # train first 160 data points
X_test, Y_test = X[160:], Y[160:]  # test remaining 40 data points
