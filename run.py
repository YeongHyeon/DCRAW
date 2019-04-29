import argparse

import tensorflow as tf

import source.neuralnet_sigentropy as nn
import source.datamanager as dman
import source.tf_process as tfp

def main():

    dataset = dman.DataSet()
    neuralnet = nn.DRAW(height=dataset.height, width=dataset.width, sequence_length=FLAGS.seqlen, learning_rate=FLAGS.lr, attention=FLAGS.attention)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, canvas_size=FLAGS.canvas, sequence_length=FLAGS.seqlen, print_step=1)
    tfp.validation(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, canvas_size=FLAGS.canvas)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch for training')
    parser.add_argument('--batch', type=int, default=1100, help='Batch size for training.')
    parser.add_argument('--canvas', type=int, default=10, help='Canvas size for generating result.')
    parser.add_argument('--seqlen', type=int, default=10, help='Sequence length of RNN.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimization.')
    parser.add_argument('--attention', type=bool, default=False, help='Use attention or not.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
