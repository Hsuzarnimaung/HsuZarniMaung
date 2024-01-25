import tensorflow as tf

from keras import layers, Input
from keras.layers import LSTMCell
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def Model(images, ht, ct):
    input = tf.compat.v1.to_float(images) / 255.0
    # conv1=(input)
    conv1 = layers.Conv2D(16, 8, 4, activation="relu",name="conv1")(input)
    conv2 = layers.Conv2D(32, 4, 2, activation="relu",name="conv2")(conv1)
    flat = layers.Flatten()(conv2)
    fully_connected = layers.Dense(256, activation="relu",name="fully_connect")(flat)
    lstm = LSTMCell(256)
    lstm_out, (h, c) = lstm(fully_connected, states=[ht, ct])

    return lstm_out, [h,c]


class Network():
    def __init__(self, output_num, reg=0.01):


            self.num_of_output = output_num
            self.states = Input(shape=[84, 84, 4], dtype=tf.uint8, name="state")
            self.ht = Input(shape=[256], dtype=tf.float32)
            self.ct = Input(shape=[256], dtype=tf.float32)
            self.total = Input(shape=[], dtype=tf.float32, name="loss")

            lstm, self.hc = Model(self.states, self.ht, self.ct)

            self.output = layers.Dense(self.num_of_output, activation=None, name="policy")(lstm)
            self.vhat = layers.Dense(1, activation=None, name="value")(lstm)
            self.vhat = tf.compat.v1.squeeze(self.vhat, squeeze_dims=[1])
            cdist = tf.compat.v1.distributions.Categorical(logits=self.output)
            self.sample_action = cdist.sample()


            with tf.compat.v1.variable_scope("network_loss"):

                self.trainer2 = tf.compat.v1.train.RMSPropOptimizer(0.0001, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.trainer2.compute_gradients(self.total)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


def Create_Network(num_of_output):
    return Network(num_of_output)


