# @Time : 2022/11/22 13:21 
# @Author : zhongyu 
# @File : train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import TensorSpec
from tensorflow import keras
from util import data
import json
from cnn import CNN
from tensorflow.keras.callbacks import EarlyStopping
import math

if __name__ == '__main__':
    east_data_dir = '../data/'
    east_config_file = './config/local_config.json'
    with open(east_config_file) as json_file:
        config = json.load(json_file)

    # shots, tags = config['shots'], config['tags']
    # da, db = data.process_path(os.path.join(east_data_dir, 'label_train', '54000', '54006.hdf5'))
    # east_train = data.concat_dataset_from_directory(os.path.join(east_data_dir, 'label_train'))
    model = CNN()
    train = tf.data.experimental.load(os.path.join(east_data_dir, 'dataset', 'train'))
    val = tf.data.experimental.load(os.path.join(east_data_dir, 'dataset', 'val'))
    val = val.batch(20)
    train = train.batch(20)

    # input2_shape = (10, 9)
    # x2 = tf.random.normal(input2_shape)
    # input1_shape = (10, 16)
    # x1 = tf.random.normal(input1_shape)
    # x = {"input_1": x1, "input_2": x2}
    # # input3 = (4,4,64)
    # # x3 = tf.random.normal(input3)
    # # y = tf.keras.layers.Reshape((64*4,))(x3)
    # y = model(x)
    # print(y.shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['BinaryAccuracy'])

    model.fit(train, batch_size=20, epochs=10, callbacks=[early_stopping], validation_data=val)
    model.summary()
    # model.save('et1')
