from typing import List
from mpi4py import MPI
import numpy as np
from mnist import Model

class SyncWorker(Model):
    def __init__(self, comm, rank, batch_size):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.batch_size = batch_size

    def work(self) -> List:
        x_batch, y_batch = self.data.train.next_batch(self.batch_size)
        # Compute gradient values
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # ret -> ((gw, w), (gb, b))
        grads = [grad for grad, var in ret]
        self.sess.run(self.grads_and_vars, feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # Tuple: (gradient, variable)
        # Pack gradeint values
        # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
        # Send data to parameter server
        return grads
