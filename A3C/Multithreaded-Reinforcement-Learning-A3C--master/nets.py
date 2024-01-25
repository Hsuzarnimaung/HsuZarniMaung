import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTMCell
def build_feature_extractor(input_):

    # scale inputs from 0-255 to 0-1
    input_ = tf.to_float(input_) / 255.0

    # CNN layers
    conv1 = tf.contrib.layers.conv2d(
        input_,
        16, # output features maps
        8, # kernel size
        4, # stride
        activation_fn=tf.nn.relu,
        scope="conv1")

    conv2 = tf.contrib.layers.conv2d(
        conv1,
        32, # output features maps
        4, # kernel size
        2, # stride
        activation_fn=tf.nn.relu,
        scope="conv2"
    )

    # image to feature vector
    flat = tf.contrib.layers.flatten(conv2)

    # dense layer (fully connected)
    fc1 = tf.contrib.layers.fully_connected(
        inputs=flat,
        num_outputs=256,
        scope="fc1")

    #lstm = LSTMCell(256)
    #state=[ht, ct]
    #lstm_out, (ht,ct)= lstm(fc1, states=[ht,ct])
    return fc1



class PolicyNetwork:
    def __init__(self, num_outputs, reg=0.01):
        self.ht = tf.zeros((1,256))
        self.ct = tf.zeros((1,256))
        self.num_outputs = num_outputs
        #self.ht_p = tf.placeholder(shape=[1, 256], dtype=tf.float32)
        #self.ct_p = tf.placeholder(shape=[1, 256], dtype=tf.float32)
        # Graph inputs
        # after resizing we have 4 consecutive frames of 84x84 size
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # Advantage = G - V(s)
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # selected actions
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        with tf.variable_scope("shared", reuse=False):
            fc1 = build_feature_extractor(self.states)
            lstm = LSTMCell(256)
            lstm_out, (self.ht, self.ct) = lstm(fc1, states=[self.ht, self.ct])
        with tf.variable_scope("policy_network"):


            self.logits = tf.contrib.layers.fully_connected(lstm_out, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits)
            self.probs = tf.clip_by_value(self.probs, 1e-20, 1.0)
            # Sample an action
            cdist = tf.distributions.Categorical(logits=self.logits)
            self.sample_action = cdist.sample()

            # Add regularization to increase exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), axis=1)

            # Get the predictions for the chosen actions only
            batch_size = tf.shape(self.states)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.loss = tf.log(self.selected_action_probs) * self.advantage + reg * self.entropy
            self.loss = -tf.reduce_sum(self.loss, name="loss")

            # training
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

            # we'll need these later for running gradient descent steps
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


class ValueNetwork:
    def __init__(self):
        self.ht = tf.zeros((1,256))
        self.ct = tf.zeros((1,256))
        #self.ht_v = tf.placeholder(shape=[1, 256], dtype=tf.float32)
        #self.ct_v = tf.placeholder(shape=[1, 256], dtype=tf.float32)
        # Graph inputs
        # after resizing we have 4 consecutive frames of 84x84 size
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        with tf.variable_scope("shared", reuse=True):
            fc1 = build_feature_extractor(self.states)
            lstm = LSTMCell(256)
            lstm_out, (self.ht, self.ct) = lstm(fc1, states=[self.ht, self.ct])
        with tf.variable_scope("value_network"):


            self.vhat = tf.contrib.layers.fully_connected(
                inputs=lstm_out,
                num_outputs=1,
                activation_fn=None
            )
            self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1], name="vhat")

            self.loss = tf.squared_difference(self.vhat, self.targets)
            self.loss = tf.reduce_sum(self.loss, name="loss")

            # training
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

            # we'll need these later for running gradient descent steps
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


# Use this to create networks, to ensure they are created in the correct order
def create_networks(num_outputs):
    policy_network = PolicyNetwork(num_outputs=num_outputs)
    value_network = ValueNetwork()
    return policy_network, value_network







