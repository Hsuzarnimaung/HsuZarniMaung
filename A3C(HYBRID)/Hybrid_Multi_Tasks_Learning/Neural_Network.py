import tensorflow as tf

from keras import layers,Input
from keras.layers import LSTMCell
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
def Model(images, ht, ct):
    input=tf.compat.v1.to_float(images)/255.0
    #conv1=(input)
    conv1=layers.Conv2D(16,8,4,activation="relu",name="conv1")(input)
    conv2=layers.Conv2D(32,4,2,activation="relu",name="conv2")(conv1)
    flat=layers.Flatten()(conv2)
    fully_connected=layers.Dense(256,activation="relu")(flat)
    lstm = LSTMCell(256)
    lstm_out, (ht, ct) = lstm(fully_connected, states=[ht, ct])
    return lstm_out, [ht, ct]

class Network():
    def __init__(self,output_num,reg=0.01):
        self.num_of_output = output_num
        self.states = Input(shape=[84, 84, 4], dtype=tf.uint8,name="X")
        self.advantages = Input(shape=[], dtype=tf.float32,name="y")
        self.actions = Input(shape=[], dtype=tf.int32,name="actions")
        self.ht = Input(shape=[ 256],dtype=tf.float32)
        self.ct = Input(shape=[256],dtype=tf.float32)


        with tf.compat.v1.variable_scope("share",reuse=False):
            lstm, state_out = Model(self.states, self.ht, self.ct)
            self.state_out = state_out
        with tf.compat.v1.variable_scope("Policy_Network"):

            # self.hct_init = [np.zeros((1, 256)), np.zeros((1, 256))]


            self.output = layers.Dense(self.num_of_output, activation=None)(lstm)

            self.probilitys=tf.nn.softmax(self.output)
            self.probility = tf.clip_by_value(self.probilitys, 1e-20, 1.0)
            cdist = tf.compat.v1.distributions.Categorical(logits=self.output)
            self.sample_action = cdist.sample()
            self.entropy = -tf.reduce_sum(self.probility * tf.math.log(self.probility), axis=1)
            batch_size = tf.shape(self.states)[0]

            gather_indices = tf.range(batch_size) * tf.shape(self.probility)[1] + self.actions
            self.selected_action_probs = tf.gather(tf.reshape(self.probility, [-1]), gather_indices)
            self.loss = tf.math.log(self.selected_action_probs) * self.advantages+reg * self.entropy
            self.loss = -tf.reduce_sum(self.loss,name="policy_loss")
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var]for grad, var in self.grads_and_vars if grad is not None]

class value_Network():
    def __init__(self):

            # self.hct_init = [np.zeros((1, 256)), np.zeros((1, 256))]

        self.vstates = Input(shape=[84, 84, 4], dtype=tf.uint8,name="X")
        self.targets = Input(shape=[], dtype=tf.float32,name="y")
        self.vht = Input(shape=[256], dtype=tf.float32)
        self.vct = Input(shape=[256], dtype=tf.float32)

        with tf.compat.v1.variable_scope("share",reuse=True):
            vlstm, vstate_out = Model(self.vstates, self.vht, self.vct)
            #self.vstate_out = [va, vb]
            self.vstate_out = vstate_out
        with tf.compat.v1.variable_scope("Value_Network"):

            self.vhat = layers.Dense(1, activation=None)(vlstm)

            self.vhat = tf.compat.v1.squeeze(self.vhat, squeeze_dims=[1], name="vhat")

            self.vloss = tf.compat.v1.squared_difference(self.vhat, self.targets)
            self.vloss = tf.reduce_sum(self.vloss, name="value_loss")
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.grads_and_vars = self.optimizer.compute_gradients(self.vloss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

def Create_Network(num_of_output):
    policy=Network(num_of_output)
    value=value_Network()
    return policy,value


"""def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer"""

