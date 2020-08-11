from typing import List
from mnist import Model
from mpi4py import MPI
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow.compat.v1 as tf
import time
from worker import SyncWorker
import logging

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    training_time = 1
    batch_size = 128
    
    comm = MPI.COMM_WORLD
    worker_index = comm.Get_rank()
    worker_size = comm.Get_size()
    
    start = time.time()
    w = SyncWorker
    
    if worker_index == 0:
        logging.info('initialize worker as {}'.format(w))
        
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    model = Model(dataset)
    
    worker = w(model=model, dataset=dataset)
    
    if worker_index != 0:
        print('shape', worker.shape)
    '''
    worker.syn_weights(worker.model.variables.get_flat())
    
    if worker_index == 0:
        logging.info("Iteration, time, loss, accuracy")
    
    i = 0
    while i <= training_time:
        if i % 10 == 0:
            loss, accuracy = worker.compute_loss_accuracy()
            if worker_index == 0:
                logging.infO("%d, %.3f, %.3f, %.3f" % (i, time.time() - start, loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
    '''
