import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
import numpy as np


def build_feature_extractor(input_, h, c):
    # scale inputs from 0-255 to 0-1
    input_ = tf.to_float(input_) / 255.0

    # CNN layers
    conv1 = tf.contrib.layers.conv2d(
        input_,
        16,  # output features maps
        8,  # kernel size
        4,  # stride
        activation_fn=tf.nn.relu,
        scope="conv1")

    conv2 = tf.contrib.layers.conv2d(
        conv1,
        32,  # output features maps
        4,  # kernel size
        2,  # stride
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

    lstm = tf.keras.layers.LSTM(256, return_sequences=True)
    rnn_out = lstm(fc1)
    return rnn_out


class Network:
    def __init__(self, num_outputs, reg=0.01):
        self.num_outputs = num_outputs
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        # Graph inputs
        # after resizing we have 4 consecutive frames of 84x84 size
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8)
        # Advantage = G - V(s)
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        # selected actions
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32)

        self.h = tf.placeholder(shape=[None, 256], dtype=tf.float32)
        self.c = tf.placeholder(shape=[None, 256], dtype=tf.float32)



        rnn_out= build_feature_extractor(self.states, self.h, self.c)




        #with tf.variable_scope("policy"):
        self.logits = tf.contrib.layers.fully_connected(rnn_out, num_outputs,weights_initializer=normalized_columns_initializer(0.01), activation_fn=None)
        cdist = tf.distributions.Categorical(logits=self.logits)
        self.sample_action = cdist.sample()
        self.probs = tf.nn.softmax(self.logits)
        self.vhat = tf.contrib.layers.fully_connected(
                inputs=rnn_out,
                num_outputs=1,
            weights_initializer=normalized_columns_initializer(1.0),
                activation_fn=None
            )
        self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1])
            # Sample an action

            # Add regularization to increase exploration
        self.entropy = -tf.reduce_sum(self.probs * tf.log(tf.maximum(self.probs, 1e-20)), axis=1)

            # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
        self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

        #self.loss =
        self.loss = -tf.reduce_sum((tf.log(tf.maximum(self.selected_action_probs, 1e-20)) * self.advantage))

        #self.v_loss =
        self.v_loss = 0.5*tf.reduce_sum(tf.squared_difference(self.vhat, self.targets))
        self.total_loss=0.5*self.v_loss+self.loss-reg * self.entropy
            # training
            # we'll need these later for running gradient descent steps
        self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]



# Use this to create networks, to ensure they are created in the correct order
def create_networks(num_outputs):
    network = Network(num_outputs=num_outputs)

    return network


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer




