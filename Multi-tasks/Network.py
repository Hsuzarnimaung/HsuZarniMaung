import tensorflow as tf
from keras import layers,Input
from keras.layers import LSTMCell
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
def Model(images, ht, ct):
    input = tf.compat.v1.to_float(images) / 255.0
    # conv1=(input)
    conv1 = layers.Conv2D(16, 8, 4, activation="relu")(input)
    conv2 = layers.Conv2D(32, 4, 2, activation="relu")(conv1)
    flat = layers.Flatten()(conv2)
    fully_connected = layers.Dense(256, activation="relu")(flat)
    lstm = LSTMCell(256)
    lstm_out, (h, c) = lstm(fully_connected, states=[ht, ct])
    return lstm_out, h, c
class Policy_Network():
    def __init__(self,num_of_action,reg=0.01):
        self.num_of_action=num_of_action
        self.ht=tf.zeros((1,256))
        self.ct=tf.zeros((1,256))

        self.states=Input(shape=[84,84,4],dtype=tf.uint8,name="x")

        self.advantage = Input(shape=[], dtype=tf.float32, name="y")
        self.actions = Input(shape=[], dtype=tf.int32, name="actions")
        with tf.compat.v1.variable_scope("shared",reuse=False):
            lstm_out, self.ht, self.ct = Model(self.states, self.ht, self.ct)
        with tf.compat.v1.variable_scope("Policy"):
            self.full_connect=layers.Dense(self.num_of_action,activation=None)(lstm_out)

            cdist=tf.compat.v1.distributions.Categorical(logits=self.full_connect)
            self.sample_action=cdist.sample()
            self.pros = tf.nn.softmax(self.full_connect)
            self.pros = tf.clip_by_value(self.pros, 1e-20, 1)
            self.entropy=-tf.reduce_sum(self.pros*tf.math.log(self.pros),axis=1)

            batch_size=tf.shape(self.states)[0]
            gather_indices=tf.range(batch_size)*tf.shape(self.pros)[1]+self.actions
            self.action_probilitys=tf.gather(tf.reshape(self.pros,[-1]),gather_indices)
            self.loss=tf.math.log(self.action_probilitys)*self.advantage+self.entropy*reg
            self.loss=-tf.reduce_sum(self.loss,name="policy_loss")
            self.trainer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.gar_and_var=self.trainer.compute_gradients(self.loss)
            self.gar_and_var=[[gar,var] for gar,var in self.gar_and_var if gar is not None]

class Value_Network():
    def __init__(self):
        self.ht=tf.zeros((1,256))
        self.ct=tf.zeros((1,256))
        self.states=Input(shape=[84,84,4],dtype=tf.uint8,name="x")
        self.targets = Input(shape=[], dtype=tf.float32, name="y")
        with tf.compat.v1.variable_scope("shared",reuse=True):
            lstm_out, self.ht, self.ct = Model(self.states, self.ht, self.ct)
        with tf.compat.v1.variable_scope("Value"):
            self.vhat=layers.Dense(1,activation=None)(lstm_out)
            self.vhat=tf.compat.v1.squeeze(self.vhat,squeeze_dims=[1],name="vhat")

            self.loss=tf.compat.v1.squared_difference(self.vhat,self.targets)
            self.loss=tf.reduce_sum(self.loss,name="value_loss")
            self.trainer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.gar_and_var=self.trainer.compute_gradients(self.loss)
            self.gar_and_var=[[gar,var] for gar,var in self.gar_and_var if gar is not None]
def Network(num_of_actions):
    policy=Policy_Network(num_of_actions)
    value=Value_Network()
    return policy, value

