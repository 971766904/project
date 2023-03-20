#EAST实时破裂预测
##脚本功能
1. shotlist.py用于读取训练、验证、测试数据集的炮号，存储为json文件
##错误解决
1. processors.py中clipprocessor可能会出现长度不一致的问题，可能来源于resample过程中丢失数据点的问题，解决方法是先进行clip再进行resample。
2. layers.Reshape()将二维或三维数据转为一维，需要在一维长度后加 **","** `c = layers.Reshape((p2.shape[1]*p2.shape[2],))(p2)`