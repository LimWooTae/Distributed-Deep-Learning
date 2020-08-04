from typing import List
from mnist import Model
#import Model
from mpi4py import MPI
import numpy as np
import tensorflow as tf
import time

class SyncWorker(Model):
    def __init__(self, comm, rank, batch_size):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.batch_size = batch_size
        self.app = self.optimizer.apply_gradients(self.grads)
        
        self.temp0 = tf.zeros(shape = self.w_conv1.shape)
        self.temp1 = tf.zeros(shape = self.b_conv1.shape)
        self.temp2 = tf.zeros(shape = self.w_conv2.shape)
        self.temp3 = tf.zeros(shape = self.b_conv2.shape)
        self.temp4 = tf.zeros(shape = self.w_fc1.shape)
        self.temp5 = tf.zeros(shape = self.b_fc1.shape)
        self.temp6 = tf.zeros(shape = self.w_fc2.shape)
        self.temp7 = tf.zeros(shape = self.b_fc2.shape)

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
        
    def apply(self): # 값은 바뀐다.
        
        self.w_conv1 = self.temp0
        self.b_conv1 = self.temp1
        self.w_conv2 = self.temp2
        self.b_conv2 = self.temp3
        self.w_fc1 = self.temp4
        self.b_fc1 = self.temp5
        self.w_fc2 = self.temp6
        self.b_fc2 = self.temp7
            
class ParameterServer(Model):
    def __init__(self, comm, rank):
        super().__init__()
        self.comm = comm
        self.rank = rank
        
    def sync(self) -> List:
        # Receive data from workers
        # From worker1
        comm.Recv([self.d1_conv1_gw, MPI.DOUBLE], source=1, tag=1)
        comm.Recv([self.d1_conv1_gb, MPI.DOUBLE], source=1, tag=2)
        comm.Recv([self.d1_conv2_gw, MPI.DOUBLE], source=1, tag=3)
        comm.Recv([self.d1_conv2_gb, MPI.DOUBLE], source=1, tag=4)
        comm.Recv([self.d1_fc1_gw, MPI.DOUBLE], source=1, tag=5)
        comm.Recv([self.d1_fc1_gb, MPI.DOUBLE], source=1, tag=6)
        comm.Recv([self.d1_fc2_gw, MPI.DOUBLE], source=1, tag=7)
        comm.Recv([self.d1_fc2_gb, MPI.DOUBLE], source=1, tag=8)

        # From worker2
        comm.Recv([self.d2_conv1_gw, MPI.DOUBLE], source=2, tag=1)
        comm.Recv([self.d2_conv1_gb, MPI.DOUBLE], source=2, tag=2)
        comm.Recv([self.d2_conv2_gw, MPI.DOUBLE], source=2, tag=3)
        comm.Recv([self.d2_conv2_gb, MPI.DOUBLE], source=2, tag=4)
        comm.Recv([self.d2_fc1_gw, MPI.DOUBLE], source=2, tag=5)
        comm.Recv([self.d2_fc1_gb, MPI.DOUBLE], source=2, tag=6)
        comm.Recv([self.d2_fc2_gw, MPI.DOUBLE], source=2, tag=7)
        comm.Recv([self.d2_fc2_gb, MPI.DOUBLE], source=2, tag=8)
        
        self.conv1_gw = (self.d1_conv1_gw + self.d2_conv1_gw)/2
        self.conv1_gb = (self.d1_conv1_gb + self.d2_conv1_gb)/2
        self.conv2_gw = (self.d1_conv2_gw + self.d2_conv2_gw)/2
        self.conv2_gb = (self.d1_conv2_gb + self.d2_conv2_gb)/2
        self.fc1_gw = (self.d1_fc1_gw + self.d2_fc1_gw)/2
        self.fc1_gb = (self.d1_fc1_gb + self.d2_fc1_gb)/2
        self.fc2_gw = (self.d1_fc2_gw + self.d2_fc2_gw)/2
        self.fc2_gb = (self.d1_fc2_gb + self.d2_fc2_gb)/2
        
        return [
            self.conv1_gw, self.conv1_gb, self.conv2_gw, self.conv2_gb,
            self.fc1_gw, self.fc1_gb, self.fc2_gw, self.fc2_gb
        ]

if __name__ == "__main__":
    training_time = 200
    batch_size = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        ps = ParameterServer(comm, rank)
    if rank == 1:
        w1 = SyncWorker(comm, rank, batch_size)
    if rank == 2:
        w2 = SyncWorker(comm, rank, batch_size)
    
    p = Model()
    conv1_gw = np.empty(p.w_conv1.shape, dtype=np.float32)
    conv1_gb = np.empty(p.b_conv1.shape, dtype=np.float32)
    conv2_gw = np.empty(p.w_conv2.shape, dtype=np.float32)
    conv2_gb = np.empty(p.b_conv2.shape, dtype=np.float32)
    fc1_gw = np.empty(p.w_fc1.shape, dtype=np.float32)
    fc1_gb = np.empty(p.b_fc1.shape, dtype=np.float32)
    fc2_gw = np.empty(p.w_fc2.shape, dtype=np.float32)
    fc2_gb = np.empty(p.b_fc2.shape, dtype=np.float32)
    
    # Mapping variables
    vars = [
        conv1_gw, conv1_gb, conv2_gw, conv2_gb,
        fc1_gw, fc1_gb, fc2_gw, fc2_gb
    ]
    
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
            temp = w1.w_conv1
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
    
