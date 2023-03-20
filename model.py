import tensorflow as tf
from tensorflow.keras import layers


class ParallelConv1D(layers.Layer):
    def __init__(self, filters=16, kernel_size=5, padding='same', activation=None):
        super(ParallelConv1D, self).__init__()
        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._activation = activation
        self.layers = []

    def build(self, input_shape):
        """

        :param input_shape: batch_size + (lstm_time_step, channels, length)
        :return:
        """
        for i in range(input_shape[2]):
            cnn_layer = layers.Conv1D(filters=self._filters, kernel_size=self._kernel_size, padding=self._padding,
                                      activation=self._activation, kernel_initializer=tf.random_normal_initializer())
            pooling_layer = layers.MaxPooling1D(pool_size=2, strides=None, padding='same')
            self.layers.append([cnn_layer, pooling_layer])

    def call(self, inputs, **kwargs):
        """

        :param inputs: batch_size + (lstm_time_step, channels, length)
        :param kwargs:
        :return: batch_size + (lstm_time_step, channels, new_steps, filters)
        """
        outputs = []
        for t in range(inputs.shape[1]):
            time_step_output = []
            step_inputs = inputs[:, t]
            for i in range(len(self.layers)):
                cnn_layer1_out = self.layers[i][0](step_inputs[:, i])
                pooling_layer_1_out = self.layers[i][1](cnn_layer1_out)
                time_step_output.append(pooling_layer_1_out)
            time_step_output = tf.stack(time_step_output)
            outputs.append(time_step_output)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [2, 0, 1, 3, 4])
        return outputs


class HybridNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c10 = ParallelConv1D(activation='relu', filters=16, kernel_size=3)
        self.c11 = ParallelConv1D(activation='relu', filters=16, kernel_size=3)
        self.c12 = ParallelConv1D(activation='relu', filters=16, kernel_size=3)
        self.c20 = ParallelConv1D(activation='relu', filters=32, kernel_size=3)
        self.c21 = ParallelConv1D(activation='relu', filters=32, kernel_size=3)
        self.c22 = ParallelConv1D(activation='relu', filters=32, kernel_size=3)
        self.c30 = ParallelConv1D(activation='relu', filters=64, kernel_size=3)
        self.c31 = ParallelConv1D(activation='relu', filters=64, kernel_size=3)
        self.c32 = ParallelConv1D(activation='relu', filters=64, kernel_size=3)
        self.c40 = ParallelConv1D(activation='relu', filters=128, kernel_size=3)
        self.c41 = ParallelConv1D(activation='relu', filters=128, kernel_size=3)
        self.c42 = ParallelConv1D(activation='relu', filters=128, kernel_size=3)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.l1 = layers.LSTM(32, activation='relu', return_sequences=True)
        self.l2 = layers.LSTM(32, activation='relu', return_sequences=False)
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(64, activation='relu')
        self.d3 = layers.Dense(16, activation='relu')
        self.d4 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        conv_inputs = inputs["input_1"]
        basic_inputs = inputs["input_2"]
        c0 = layers.Reshape(target_shape=(conv_inputs.shape[1], conv_inputs.shape[2], conv_inputs.shape[3], 1))(conv_inputs)
        l0 = layers.Reshape(target_shape=(basic_inputs.shape[1], basic_inputs.shape[2]))(basic_inputs)
        c10 = self.c10(c0)
        c11 = self.c11(c10)
        c12 = self.c12(c11)
        c20 = self.c20(c12)
        c21 = self.c21(c20)
        c22 = self.c22(c21)
        c30 = self.c30(c22)
        c31 = self.c31(c30)
        c32 = self.c32(c31)
        c40 = self.c40(c32)
        c41 = self.c41(c40)
        c42 = self.c42(c41)
        c = layers.Reshape((c42.shape[1], c42.shape[2]*c42.shape[3]*c42.shape[4]))(c42)
        bn1 = self.bn1(c)
        concat = layers.Concatenate()([bn1, l0])
        l1 = self.l1(concat)
        l2 = self.l2(l1)
        bn2 = self.bn2(l2)
        d1 = self.d1(bn2)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        return d4
