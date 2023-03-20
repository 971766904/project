# @Time : 2022/11/30 13:42 
# @Author : zhongyu 
# @File : frozen_graph.py

# frozen_graph.py
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

loaded = tf.saved_model.load('et1')
infer = loaded.signatures['serving_default']

f = tf.function(infer).get_concrete_function(input_1=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32))
f2 = convert_variables_to_constants_v2(f)
graph_def = f2.graph.as_graph_def()

# Export frozen graph
with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())
