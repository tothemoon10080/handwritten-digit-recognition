import tensorflow as tf
import keras
from keras import datasets, layers, models
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np

#设置GPU
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0],"GPU")#仅使用第0个GPU

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()#加载mnist数据集(在线下载)

train_images = train_images.reshape((60000, 28, 28, 1))#将训练集的数据形状从(60000, 28, 28)转换为(60000, 28, 28, 1) 灰色图像
test_images = test_images.reshape((10000, 28, 28, 1))#将测试集的数据形状从(10000, 28, 28)转换为(10000, 28, 28, 1) 灰色图像
train_images, test_images = train_images / 255.0, test_images / 255.0#将训练集和测试集的数据归一化到0~1之间
train_labels = keras.utils.to_categorical(train_labels, 10) 

model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))#卷积层
model.add(layers.MaxPooling2D((2, 2)))#池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))#卷积层
model.add(layers.MaxPooling2D((2, 2)))#池化层
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())#展平
model.add(layers.Dense(64, activation='relu'))#全连接层
model.add(layers.Dense(10, activation="softmax"))#全连接层

model.compile(
    # 设置优化器为Adam优化器
    optimizer='adam',
    # 设置损失函数为交叉熵损失函数（tf.keras.losses.SparseCategoricalCrossentropy()）
    # from_logits为True时，会将y_pred转化为概率（用softmax），否则不进行转换，通常情况下用True结果更稳定
    loss=tf.keras.losses.CategoricalCrossentropy(),
    # 设置性能指标列表，将在模型训练时监控列表中的指标
    metrics=['accuracy'])

history = model.fit(
    train_images, 
    train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels))#训练10轮

pre = model.predict(test_images)
print(test_labels[1])
print(np.argmax(pre[1]))