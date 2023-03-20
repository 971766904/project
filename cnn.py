# @Time : 2022/11/21 16:42 
# @Author : zhongyu 
# @File : cnn.py.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        conv_inputs = inputs["input_1"]
        basic_inputs = inputs["input_2"]
        c0 = layers.Reshape(target_shape=(conv_inputs.shape[1], 1))(conv_inputs)
        # l0 = layers.Reshape(target_shape=(basic_inputs.shape[1]))(basic_inputs)
        print(c0.shape)
        c1 = self.conv1(c0)
        print(c1.shape)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c = layers.Reshape((p2.shape[1]*p2.shape[2],))(p2)
        x = layers.Concatenate()([c, basic_inputs])
        x = self.dense1(x)  # [batch_size, 1024]
        output = self.dense2(x)  # [batch_size, 1]

        return output
