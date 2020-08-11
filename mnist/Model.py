from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Model:
    def __init__(self):
        # Data: mnist dataset
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        # CNN model
        with tf.variable_scope("mnist", reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.x_image = tf.reshape(self.x, [-1,28,28,1])
            self.y_ = tf.placeholder(tf.float32, [None, 10])

            '''First Conv layer'''
            # shape: [5,5,1,32]
            self.w_conv1 = tf.get_variable("v0", shape=[5,5,1,32], dtype=tf.float32)
            # shape: [32]
            self.b_conv1 = tf.get_variable("v1", shape=[32], dtype=tf.float32)
            # conv layer
            self.conv1 = tf.nn.conv2d(self.x_image, self.w_conv1, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)
            self.h_pool1 = tf.nn.max_pool2d(self.h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            '''Second Conv layer'''
            # shape: [5,5,32,64]
            self.w_conv2 = tf.get_variable("v2", shape=[5,5,32,64], dtype=tf.float32)
            # shape: [64]
            self.b_conv2 = tf.get_variable("v3", shape=[64], dtype=tf.float32)
            # conv layer
            self.conv2 = tf.nn.conv2d(self.h_pool1, self.w_conv2, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
            self.h_pool2 = tf.nn.max_pool2d(self.h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
            '''FC layer'''
            self.w_fc1 = tf.get_variable("v4", shape=[7*7*64, 1024], dtype=tf.float32)
            self.b_fc1 = tf.get_variable("v5", shape=[1024], dtype=tf.float32)
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)

            '''Dropout'''
            self.keep_prob = tf.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, rate=1.0-self.keep_prob)

            '''Softmax layer'''
            self.w_fc2 = tf.get_variable("v6", shape=[1024, 10], dtype=tf.float32)
            self.b_fc2 = tf.get_variable("v7", shape=[10], dtype=tf.float32)
            self.logits = tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2
            self.y = tf.nn.softmax(self.logits)

            '''Cost function & optimizer'''
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer(1e-4)
        
            # Variables
            self.var_size = 8
            self.var_shape = [
                [5,5,1,32],
                [32],
                [5,5,32,64],
                [64],
                [7*7*64, 1024],
                [1024],
                [1024, 10],
                [10]
            ]
            self.var_bucket = [tf.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]

            # For evaluating
            self.prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
            self.test_x = self.data.test.images
            self.test_y_ = self.data.test.labels
            self.train_step = self.optimizer.minimize(self.cost)
            
            # Create session
            self.sess = tf.Session()

            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

            # Gradients
            self.grads = self.optimizer.compute_gradients(self.cost, self.var_bucket)
            self.grads_placeholder = [
                (tf.placeholder("float", shape = grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(self.grads_placeholder)

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads], feed_dict = {self.x: x, self.y_: y, self.keep_prob: 0.5})
        
    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
            
        self.sess.run(self.apply_grads_placeholder, feed_dict = feed_dict)
        
    def compute_loss_accuracy(self, x, y):
        return self.sess.run([self.cost, self.accuracy], feed_dict = {self.x: x, self.y_ : y, self.keep_prob : 1.0})

