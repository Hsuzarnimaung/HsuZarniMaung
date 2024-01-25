import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
import numpy as np


class Network:
    def __init__(self, num_outputs, scope, reg=0.01):
        self.scope = scope
        self.num_outputs = num_outputs
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        with tf.variable_scope(self.scope):
            # Graph inputs
            # after resizing we have 4 consecutive frames of 84x84 size
            self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="states")

            self.h = tf.placeholder(shape=[None, 256], dtype=tf.float32)
            self.c = tf.placeholder(shape=[None, 256], dtype=tf.float32)
            # with tf.variable_scope("share"):
            input_ = tf.to_float(self.states) / 255.0

            # CNN layers
            conv1 = tf.contrib.layers.conv2d(
                input_,
                16,  # output features maps
                8,  # kernel size
                4,  # stride
                activation_fn=tf.nn.relu,
            )

            conv2 = tf.contrib.layers.conv2d(
                conv1,
                32,  # output features maps
                4,  # kernel size
                2,  # stride
                activation_fn=tf.nn.relu,

            )

            # image to feature vector
            flat = tf.contrib.layers.flatten(conv2)

            # dense layer (fully connected)
            fc1 = tf.contrib.layers.fully_connected(
                inputs=flat,
                num_outputs=256,
            )
            lstm_input = tf.expand_dims(fc1, [0])

            step_size = tf.shape(self.states)[:1]
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            lstm_state_in = tf.contrib.rnn.LSTMStateTuple(self.c, self.h)

            lstm_out, lstm_state_out = tf.nn.dynamic_rnn(lstm_cell, lstm_input,
                                                         initial_state=lstm_state_in,
                                                         sequence_length=step_size,
                                                         time_major=False)

            lstm_c, lstm_h = lstm_state_out
            self.hct = [ lstm_h[:1, :],lstm_c[:1, :]]
            rnn_out = tf.reshape(lstm_out, [-1, 256])

            self.logits = tf.contrib.layers.fully_connected(inputs=rnn_out, num_outputs=num_outputs,
                                                            activation_fn=tf.nn.softmax)
            self.vhat = tf.contrib.layers.fully_connected(
                inputs=rnn_out,
                num_outputs=1,
                activation_fn=None,

            )

            if (self.scope != 'global'):
                # Advantage = G - V(s)
                self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage")
                # selected actions
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

                # Sample an action
                # Add regularization to increase exploration

                entropy = -tf.reduce_sum(self.logits * tf.log(tf.maximum(self.logits, 1e-20)), axis=1)
                # Get the predictions for the chosen actions only
                selected_action_probs = tf.gather(tf.reshape(self.logits, [-1]), self.actions)
                loss = -tf.reduce_sum((tf.log(tf.maximum(selected_action_probs, 1e-20)) * self.advantage))
                v_loss = 0.5 * tf.reduce_sum(tf.square(self.advantage))
                self.total = loss + 0.5 * v_loss - reg * entropy
                local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                gradients = tf.gradients(self.total, local_params)
                grads, grad_norms = tf.clip_by_global_norm(gradients, 5)
                master_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = self.optimizer.apply_gradients(list(zip(grads, master_net_params)))


# Use this to create networks, to ensure they are created in the correct order
def create_networks(num_outputs, scope):
    network = Network(num_outputs=num_outputs, scope=scope)

    return network







