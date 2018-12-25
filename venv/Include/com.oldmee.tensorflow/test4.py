import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
import keras
from keras.models import save_model,load_model
from keras.models import Model

#1 加载数据集，对数据集进行处理，把输入和结果进行分开
df = pd.read_csv("train.csv")
data = df.as_matrix()
df = None

#打乱顺序
np.random.shuffle(data)

x_train = data[:,1:]
#把训练的图片数据转化成28*28的图片
x_train = x_train.reshape(data.shape[0],28,28,1).astype("float32")
x_train = x_train/255
#把训练的图片进行OneHot编码
y_train = np_utils.to_categorical(data[:,0],10).astype("float32")

#2 设置相关参数
# 设置对训练集的批次大小
batch_size = 64
# 设置卷积过滤个数
n_filters = 32
# 设置最大池化，池化核大小
pool_size = (2,2)

#3 定义网络，按照ZeroPadding，卷积层、规范层、池化层进行设置
# 这里用到了最新的selu，很厉害的一种激活函数
cnn_net = Sequential()
cnn_net.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size = pool_size))

cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(48,kernel_size=(3,3)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size = pool_size))

cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(64,kernel_size=(2,2)))
cnn_net.add(Activation('relu'))
cnn_net.add(BatchNormalization(epsilon=1e-6,axis=1))
cnn_net.add(MaxPool2D(pool_size=pool_size))

cnn_net.add(Dropout(0.25))
cnn_net.add(Flatten())

cnn_net.add(Dense(3168))
cnn_net.add(Activation('relu'))

cnn_net.add(Dense(10))
cnn_net.add(Activation('softmax'))

#4 查看网络结构，可视化模型
# 查看网络结构
cnn_net.summary()

from keras.utils.vis_utils import plot_model, model_to_dot
from IPython.display import Image, SVG

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 可视化模型
SVG(model_to_dot(cnn_net).create(prog='dot',format='svg'))

#5 训练模型、保存和载入模型
cnn_net = load_model('cnn_net_model.h5')
# cnn_net.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# cnn_net.fit(x_train,y_train,batch_size=batch_size,epochs=50,verbose=1,validation_split=0.2)
# cnn_net.save('cnn_net_model.h5')

#6 生成提交的预测结果
df = pd.read_csv("test.csv")
x_valid = df.values.astype('float32')
n_valid = x_valid.shape[0]
x_valid = x_valid.reshape(n_valid,28,28,1)
x_valid = x_valid/255
yPred = cnn_net.predict_classes(x_valid,batch_size=32,verbose=1)
np.savetxt('mnist_output.csv',np.c_[range(1,len(yPred)+1),yPred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
































