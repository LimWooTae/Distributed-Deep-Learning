from typing import List
from mpi4py import MPI
import numpy as np
from mnist import Model
import logging


class SyncWorker(Model):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    
    def work(self) -> List:
        x_batch, y_batch = self.data.train.next_batch(self.batch_size)
        # Compute gradient values
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # ret -> ((gw, w), (gb, b))
        grads = [grad for grad, var in ret]

        # Tuple: (gradient, variable)
        # Pack gradeint values
        # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
        # Send data to parameter server
        return grads
    
    '''
    def syn_weights(self, weights):
        for i in range(self.worker_size):
            received = self.comm.Bcast(weights[self.shards[i]: self.shards[i+1]], root=i)
            if i != self.worker_index:
                weights[self.shards[i]: self.shards[i + 1]] = received[:]
        self.model.variables.set_flat(weights)

        
    def compute_gradients(self):
        xs, ys = self.dataset.train.next_batch(self.batch_size)
        return self.model.compute_gradients(xs, ys)
        
    def apply_gradient(self, flat_grad, recv_grad):
        self.apply_gradient_with_optimizer(flat_grad, recv_grad)
    
    def apply_gradient_with_optimizer(self, flat_grad, recv_grad):
        flat_grad[self.shards[self.worker_index]: self.shards[self.worker_index + 1]] = np.mean(recv_grad, axis=0)
        self.net.apply_gradients(tf_variables.unflatten(flat_grad, self.shapes))
        
        weights = self.net.variables.get_flat()
        self.syn_weights(weights)
        
    def shuffle_reduce(self, gradients):
        """
        1/2. send gradient shards to others
        1/2. receive gradient shard from others
        3. aggregate gradient shards
        4. send reduced gradient shard back
        :param gradients:
        :return:
        """
        flat_grad = np.concatenate([g.flatten() for g in gradients])
        recv_grad = np.empty(shape=(self.worker_size, self.local_shard_size), dtype=np.float32)
        for i in range(self.worker_size):
            sendbuf = flat_grad[self.shards[i]: self.shards[i + 1]]
            self.comm.Gather(sendbuf, recv_grad, root=i)

        self.apply_gradient(flat_grad, recv_grad)
        
    def compute_loss_accuracy(self):
        xs, ys = self.dataset.train.next_batch(self.test_size)
        return self.model.compute_loss_accuracy(xs, ys)
    '''
