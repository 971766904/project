# @Time : 2022/11/23 17:40 
# @Author : zhongyu 
# @File : tfdataset.py
from util.processors import SoftLabelProcessor, SliceBeforeLabelProcessor, ConvStackProcessor, BinaryLabelProcessor
from dataset import make_ds
from util.dpt import ShotSet, Shot
import os
import json
import tensorflow as tf


if __name__ == '__main__':
    east_data_dir = '../data/'
    tf.data.experimental.save(make_ds(os.path.join(east_data_dir, 'label_train')),
                              os.path.join(east_data_dir, 'dataset', 'train'))
    tf.data.experimental.save(make_ds(os.path.join(east_data_dir, 'label_val')),
                              os.path.join(east_data_dir, 'dataset', 'val'))
    tf.data.experimental.save(make_ds(os.path.join(east_data_dir, 'label_test')),
                              os.path.join(east_data_dir, 'dataset', 'test'))
    # datat1 = tf.data.experimental.load(os.path.join(east_data_dir, 'dataset', 'train'))
    # datat1 = datat1.batch(20)
    # print('done')
    # for elem in datat1:
    #     print(elem.shape)

