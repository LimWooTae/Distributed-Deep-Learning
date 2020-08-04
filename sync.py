from mnist import Model
from mpi4py import MPI
from typing import List
import numpy as np
import tensorflow as tf
import time,sys

class SyncWorker(Model):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def work(self) -> List:
        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            x_batch, y_batch = self.data.train.next_batch(self.batch_size)
            # Compute gradient values
            ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
            # ret, 이 epoch >= 17이면, 2회차부터 nan을 반환함, 싹 다 비워짐
            # ret -> ((gw, w), (gb, b))
            
            #'''
            if rank == 1:
                print("ret : ",ret[0]) # self.cost, var_bucket, cost가 사라진다
                print("_____________________________________________________")
                #print("var_bucket : ",self.sess.run(self.var_bucket[0]))
            #'''
        
            '''
            Problem: nan gradients
            print(ret[-1])
            '''
            
            # epoch 2부터 nan을 저장함
            #grads = [grad for grad, var in ret]
            grads = [var for grad, var in ret]
            '''
            if rank == 1:
                print("grads : ",grads[0])
            '''
            # Tuple: (gradient, variable)
            # Pack gradeint values 
            # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
            # Send data to parameter server
            return grads


class ParameterServer:
    def __init__(self):
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

        # Data for worker
        # For worker1
        self.w1_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker2
        self.w2_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker1 + worker2 
        self.w_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
    
        # TF variables
        with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
            self.var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
            
            # Optimizer
            self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

            # Apply gradients
            # Tuple: (gradient, variable)
            # Pack gradeint values 
            self.grads_and_vars = [(self.w_bucket[i], self.var_bucket[i]) for i in range(self.var_size)]
            self.sync_gradients = self.optimizer.apply_gradients(self.grads_and_vars)
            
            # Create session
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def update(self):
        for i in range(self.var_size):
            self.w_bucket[i] = self.w1_bucket[i] + self.w2_bucket[i]

        with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
            #var_bucket이 적용이 안되고 있는듯
            #print(self.grads_and_vars)
            self.sess.run(self.sync_gradients)
            

if __name__ == "__main__":
    epoch = 2 # 17로 설정하면 2번째 이터부터 그라디언트 손실됨.
    batch_size = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    
    # Parameter Server
    if rank == 0:
        ps = ParameterServer()
        # For broadcasting
        bucket = [np.empty(ps.var_shape[i], dtype=np.float32) for i in range(ps.var_size)]

        for step in range(epoch):
            # Receive data from workers
            # From worker1
            for i in range(ps.var_size):
                comm.Recv([ps.w1_bucket[i], MPI.DOUBLE], source=1, tag=i+1)
            print("bcast1 _____________________________________________________")
            # From worker2
            for i in range(ps.var_size):
                comm.Recv([ps.w2_bucket[i], MPI.DOUBLE], source=2, tag=i+1)
            print("bcast2 _____________________________________________________")
            '''
            Problem: nan
            print(ps.w1_bucket)
            print(ps.w2_bucket)
            '''

            # Synchronize
            ps.update()
            print("bcast3 _____________________________________________________")

            with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
                # Broadcast values
                for i in range(ps.var_size):
                    #bucket[i] = ps.var_bucket[i].eval(session=ps.sess)
                    bucket[i] = ps.w_bucket[i]
                    '''
                    #Problem: same values
                    if i==7:
                        print(bucket[i])
                        #print(ps.sess.run(ps.var_bucket[i]))
                        print("_____________________________________________________")
                    '''
                # send to worker 1, 2
            print("bcast4 _____________________________________________________")
            for i in range(ps.var_size):
                comm.Bcast([bucket[i], MPI.DOUBLE], root=0)
            print("bcast5 _____________________________________________________")


    # Worker1
    elif rank == 1:
        start = time.time()

        w1 = SyncWorker(batch_size) 
        # For broadcasting 
        bucket = [np.empty(w1.var_shape[i], dtype=np.float32) for i in range(w1.var_size)]

        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=w1.var_shape[i], dtype=tf.float32) for i in range(w1.var_size)]
            bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w1.var_size)]
            #print("var_bucket",w1.sess.run(var_bucket))
            #print("bucket",bucket)

            for step in range(epoch):
                grads_w1 = w1.work()
                #'''
                #Problem: nan
                #print("_____________________________________________________")
                #print(grads_w1)
                #'''

                # Send worker 1's grads
                for i in range(w1.var_size):
                    comm.Send([grads_w1[i], MPI.DOUBLE], dest=0, tag=i+1)
                print("bcast6 _____________________________________________________")
                # Receive data from parameter server
                for i in range(w1.var_size):
                    comm.Bcast([bucket[i], MPI.DOUBLE], root=0)
                #print(bucket[7])
                # Assign broadcasted values
                w1.sess.run(bucket_assign)
                print("bcast7 _____________________________________________________")
                #print(w1.sess.run(bucket_assign))
                
            
            end = time.time()
            print("Worker{} Accuracy: {}".format(rank,w1.sess.run(w1.accuracy, feed_dict={w1.x: w1.test_x, w1.y_: w1.test_y_, w1.keep_prob: 1.0})))
            print("Time: {}".format(end-start))
     

    # Worker2
    elif rank == 2:
        start = time.time()

        w2 = SyncWorker(batch_size) 
        # For broadcasting 
        bucket = [np.empty(w2.var_shape[i], dtype=np.float32) for i in range(w2.var_size)]

        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=w2.var_shape[i], dtype=tf.float32) for i in range(w2.var_size)]
            bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w2.var_size)]

            for step in range(epoch):
                grads_w2 = w2.work()
                '''
                Problem: nan
                print(grads_w2)
                '''
                
                # Send worker 1's grads
                for i in range(w2.var_size):
                    comm.Send([grads_w2[i], MPI.DOUBLE], dest=0, tag=i+1)
                print("bcast8 _____________________________________________________")
                # Receive data from parameter server
                for i in range(w2.var_size):
                    comm.Bcast([bucket[i], MPI.DOUBLE], root=0)

                # Assign broadcasted values
                w2.sess.run(bucket_assign)
                print("bcast9 _____________________________________________________")
            end = time.time()
            print("Worker{} Accuracy: {}".format(rank,w2.sess.run(w2.accuracy, feed_dict={w2.x: w2.test_x, w2.y_: w2.test_y_, w2.keep_prob: 1.0})))
            print("Time: {}".format(end-start))
    
    
