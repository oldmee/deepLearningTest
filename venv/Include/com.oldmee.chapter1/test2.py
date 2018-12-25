from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import test1

model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))
model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X_train, Y_train, epochs=100, batch_size=64)
