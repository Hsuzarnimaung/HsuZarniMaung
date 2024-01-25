import tensorflow as tf

from keras import layers, Input
from keras.layers import LSTMCell
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def Model(images, ht, ct):
    input = tf.compat.v1.to_float(images) / 255.0
    # conv1=(input)
    conv1 = layers.Conv2D(16, 8, 4, activation="relu", name="conv1")(input)
    conv2 = layers.Conv2D(32, 4, 2, activation="relu", name="conv2")(conv1)
    flat = layers.Flatten()(conv2)
    fully_connected = layers.Dense(256, activation="relu", name='fully')(flat)
    lstm = LSTMCell(256)
    lstm_out, (h, c) = lstm(fully_connected, states=[ht, ct])
    return lstm_out, h, c


class Network():
    def __init__(self, output_num, scope, trainer, reg=0.01):
        with tf.compat.v1.variable_scope(scope):
            self.ht = tf.zeros((1, 256))
            self.ct = tf.zeros((1, 256))
            self.num_of_output = output_num
            self.states = Input(shape=[160, 160, 4], dtype=tf.uint8, name="state")
            self.advantages = Input(shape=[], dtype=tf.float32, name="advantages")
            self.actions = Input(shape=[], dtype=tf.int32, name="actions")
            self.targets = Input(shape=[], dtype=tf.float32, name="target")

            with tf.compat.v1.variable_scope("share"):
                lstm, self.ht, self.ct = Model(self.states, self.ht, self.ct)

            with tf.compat.v1.variable_scope("policy_network"):
                self.output = layers.Dense(self.num_of_output, activation=None,name="policy")(lstm)
                cdist = tf.compat.v1.distributions.Categorical(logits=self.output)
                self.sample_action = cdist.sample()

                self.probilitys = tf.nn.softmax(self.output)
                self.probilitys = tf.clip_by_value(self.probilitys, 1e-20, 1)
                self.entropy = -tf.reduce_sum(self.probilitys * tf.math.log(self.probilitys),
                                              axis=1)
                self.selected_action_probs = tf.gather(tf.reshape(self.probilitys, [-1]), self.actions)
                self.p_loss = tf.math.log(self.selected_action_probs) * self.advantages + self.entropy * 0.01
                self.p_loss = -tf.reduce_sum(self.p_loss,name="Policy_loss")
                self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.p_loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


            with tf.compat.v1.variable_scope("value_network"):
                self.vhat = layers.Dense(1, activation=None,name="value")(lstm)

                self.vhat = tf.compat.v1.squeeze(self.vhat, squeeze_dims=[1], name="vhat")

                self.vloss = tf.compat.v1.squared_difference(self.vhat, self.targets)
                self.vloss = tf.reduce_sum(self.vloss,name="Value_loss")
                self.voptimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.vgrads_and_vars = self.voptimizer.compute_gradients(self.vloss)
                self.vgrads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]




def Create_Network(num_of_output, scope, trainer):
    return Network(num_of_output, scope=scope, trainer=trainer)


"""def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer"""
