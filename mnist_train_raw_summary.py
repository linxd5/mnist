#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import time
import os
from six.moves import xrange
import mnist_model


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epoches', 1000, 'Number of epoches to run trainer, 1000 default.')
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate, 1e-2 default.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout, 0.5 default.')
flags.DEFINE_integer('batch_size', 50, 'Batch size for SGD, 50 default.')
flags.DEFINE_integer('early_stop', 10, 'Early stop when accuracy not improve, 10 default.')
flags.DEFINE_integer('run_time', 5, 'Run time for precise result, 5 default.')
flags.DEFINE_integer('gpu', -1, 'Which gpu to run on, -1 (all gpu) default.')
flags.DEFINE_string('summaries_dir', '/tmp/mnist', 
                    'where to write event logs and checkpoint, /tmp/mnist default.')

if FLAGS.gpu != -1:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(FLAGS.gpu))



# if accuracy on validation_set not improve for 10 epoches, then stop training
best_accuracy = 0.0
best_epoch = 0

x = tf.placeholder(tf.float32, shape=[None,784])
x_image = tf.reshape(x, [-1,28,28,1])
y_ = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)
batch_loc = tf.placeholder(tf.int64)


def batch_average(dataset, method, keep_dropout):
    sum_batch = 0.0
    for j in xrange(0, dataset.num_examples, FLAGS.batch_size):
        tmp_end = np.min([j+FLAGS.batch_size, dataset.num_examples])
        sum_batch += method.eval(feed_dict={
            x: dataset.images[j:tmp_end],   
            y_: dataset.labels[j:tmp_end],
            keep_prob: keep_dropout
        })
    return sum_batch / (dataset.num_examples/FLAGS.batch_size)


def train():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    y_conv = mnist_model.inference(x_image)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # wrong_images = [mnist.test.images[batch_loc+counter] 
    #                 for counter, value in enumerate(correct_prediction) if value == False]
    # tf.image_summary('wrong_images', wrong_images)
        
    sess.run(tf.initialize_all_variables())

    # for i in range(FLAGS.max_epoches):
    for i in range(1):
        perm = np.arange(mnist.train.num_examples)
        np.random.shuffle(perm)
        temp_images = mnist.train.images[perm]
        temp_labels = mnist.train.labels[perm]
        for j in xrange(0, mnist.train.num_examples, FLAGS.batch_size):
            tmp_end = np.min([j+FLAGS.batch_size, mnist.train.num_examples])
            train_step.run(feed_dict={
                    x: temp_images[j:tmp_end],
                    y_: temp_labels[j:tmp_end],
                    keep_prob: FLAGS.dropout})

    print('training finished')

    # tf.image_summary('test', tf.reshape(mnist.test.images, [-1,28,28,1]), 10)

    wrong_images = []
    predicted_labels = []
    actual_labels = []
    for i in xrange(0, mnist.test.num_examples, FLAGS.batch_size):
        tmp_end = np.min([i+FLAGS.batch_size, mnist.test.num_examples])
        logits, temp = sess.run([y_conv, correct_prediction], feed_dict={
            x: mnist.test.images[i:tmp_end],
            y_: mnist.test.labels[i:tmp_end],
            keep_prob: 1.0})
        
        for k in xrange(len(temp)):
            if temp[k] == False:
                predicted_labels += [np.argmax(logits[k], 0)]
                actual_labels += [np.argmax(mnist.test.labels[i+k], 0)]
                wrong_images += [mnist.test.images[i+k]] 
                # print('len(predicted_labels): ', (predicted_labels))

    wrong_images = np.array(wrong_images)
    tf.image_summary('wrong_images', tf.reshape(wrong_images, [-1,28,28,1]), len(wrong_images))


    merged = tf.merge_all_summaries()
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir+'/test')


    for i in xrange(0, mnist.test.num_examples, FLAGS.batch_size):
        tmp_end = np.min([i+FLAGS.batch_size, mnist.test.num_examples])
        summary, acc = sess.run([merged, accuracy], feed_dict={
            x: mnist.test.images[i:tmp_end],
            y_: mnist.test.labels[i:tmp_end],
            keep_prob: 1.0})
        test_writer.add_summary(summary, i)


    test_accuracy = batch_average(mnist.test, accuracy, keep_dropout=1.0)
    print("test accuracy %.5f" %test_accuracy)

    for i in xrange(0, len(predicted_labels)):
        print('pic %d:   predicted_labels: %d   actual_labels: %d' 
              % (i, predicted_labels[i], actual_labels[i]))

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
