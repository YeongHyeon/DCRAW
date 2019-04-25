import os, inspect, glob, random, shutil
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self):

        self.dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.num_tr = self.dataset.train.num_examples
        self.num_te = self.dataset.test.num_examples
        print("Training: %d" %(self.num_tr))
        print("Test    : %d" %(self.num_te))
        self.height, self.width = 28, 28

    def next_train(self, batch_size):

        x, y = self.dataset.train.next_batch(batch_size)

        return x, y

    def next_test(self, batch_size):

        x, y = self.dataset.test.next_batch(batch_size)

        return x, y
