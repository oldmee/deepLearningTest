# coding=utf-8
__author__ = 'zhangxiaozi'

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

iris = datasets.load_iris()  # 得到一个字典对象，可以通过输出，以及keys,values来测试词典内容

print(type(iris))  # <class 'sklearn.utils.Bunch'>

print(iris.keys())
# dict_keys(['target', 'DESCR', 'target_names', 'feature_names', 'data'])

X = iris['data']
Y = iris['target']
print(X.shape)

# 下面可以用X和Y来分类

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.5, random_state=1)
# train_size表示训练集测试集按1:1划分
# random_state是随机数种子，为了重复试验时，划分结果是一样的，为0或不填，每次都不一样
# http://blog.csdn.net/zahuopuboss/article/details/54948181


print(train_X.shape)

LR = LogisticRegressionCV()  # 训练模型，CV表示使用了交叉验证来选择正则化系数C
LR.fit(train_X, train_Y)
# print (LR.score(test_X,test_Y))
print("Accuracy = {:.4f}".format(LR.score(test_X, test_Y)))

'''下面使用keras，神经网络进行分类，看准确你是否有提高'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))

model.add(Dense(16))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=0)

loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)

print(accuracy)
