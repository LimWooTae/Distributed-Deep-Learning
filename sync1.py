from typing import List
from mnist import Model
#import Model
from mpi4py import MPI
import numpy as np
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()

class SyncWorker(Model):
    def __init__(self, comm, rank, batch_size):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.batch_size = batch_size
        self.temp = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

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
        
    def apply(self):
        for i in range(8):
            self.layer[i] = self.temp[i]
            
class ParameterServer(Model):
    def __init__(self, comm, rank):
        super().__init__()
        self.comm = comm
        self.rank = rank
        
    def sync(self) -> List:
        # Receive data from workers
        for i in range(8):
            comm.Recv([self.d1[i], MPI.DOUBLE], source=1, tag=i+1)
        
        for i in range(8):
            comm.Recv([self.d2[i], MPI.DOUBLE], source=2, tag=i+1)
        
        for i in range(8):
            self.layer[i] = (self.d1[i] + self.d2[i])/2

        return self.layer

if __name__ == "__main__":
    training_time = 100
    batch_size = 128
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        ps = ParameterServer(comm, rank)
    if rank == 1:
        w1 = SyncWorker(comm, rank, batch_size)
    if rank == 2:
        w2 = SyncWorker(comm, rank, batch_size)
    
    p = Model()
    vars = [np.empty(p.var_shape[i], dtype=np.float32) for i in range(p.var_size)]
    
    if rank == 0:
        start = time.time()
    if rank == 1:
        w1.apply()
    elif rank == 2:
        w2.apply()

    # Measure time
    for step in range(training_time):
        # Parameter Server
        if rank == 0:
            vars = ps.sync()
        
        # Worker1
        elif rank == 1:
            grads_w1 = w1.work()
            # Send worker 1's grads
            for i in range(8):
                comm.Send([grads_w1[i], MPI.DOUBLE], dest=0, tag=i+1)
        
        # Worker2
        else:
            grads_w2 = w2.work()
            # Send worker 2's grads
            for i in range(8):
                comm.Send([grads_w2[i], MPI.DOUBLE], dest=0, tag=i+1)
    
        # Receive data from parameter server
        for i in range(8):
            comm.Bcast([vars[i], MPI.DOUBLE], root = 0)
        
        if rank == 1:
            w1.apply()
        elif rank == 2:
            w2.apply()
        
    if rank == 0:
        end = time.time()
        print('time : %0.2f'%(end - start))
    
    elif rank == 1:
        print("step : ", step, w1.sess.run(w1.accuracy, feed_dict={w1.x: w1.test_x, w1.y_: w1.test_y_, w1.keep_prob: 1.0}))
    
    elif rank == 2:
        print("step : ", step, w2.sess.run(w2.accuracy, feed_dict={w2.x: w2.test_x, w2.y_: w2.test_y_, w2.keep_prob: 1.0}))
    
