from typing import List
from mpi4py import MPI
import numpy as np

class ParameterServer(shape):
    def __init__(self):
        self.comm = comm
        self.rank = rank
        self.var_shape = shape
        slef.var_size = len(var_shape)
        
        # worker1
        self.g1 = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
        # worker2
        self.g2 = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
        # worker1 + worker2
        self.g = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
        
        # 수정 필요
        with tf.variable_scope("ParameterServer", reuse=tf.AUTO_REUSE):
            self.var = [tf.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
            
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        
        self.grads_and_vars = [(self.g[i], self.var[i]) for i in range(self.var_size)]
        self.sync_gradient = self.optimizer.apply_gradients(self.grads_and_vars)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def receive():
        for i in range(self.var_size):
            comm.Recv([self.g1[i], MPI.DOUBLE], source=1, tag=i+1)
        for i in range(self.var_size):
            comm.Recv([self.g2[i], MPI.DOUBLE], source=2, tag=i+1)
        
    def update(self):
        for i in range(self.var_size):
            self.g[i] = (self.g1[i] + self.g2[i])/2
        
        self.grads_and_vars = [(self.g[i], self.var[i]) for i in range(self.var_size)]
        self.sess.run(self.sync_gradient)
        
