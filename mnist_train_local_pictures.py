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
from PIL import Image


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epoches', 1000, 'Number of epoches to run trainer, 1000 default.')
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate, 1e-2 default.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout, 0.5 default.')
flags.DEFINE_integer('batch_size', 50, 'Batch size for SGD, 50 default.')
flags.DEFINE_integer('early_stop', 10, 'Early stop when accuracy not improve, 10 default.')
flags.DEFINE_integer('run_time', 5, 'Run time for precise result, 5 default.')
flags.DEFINE_integer('gpu', -1, 'Which gpu to run on, -1 (all gpu) default.')
flags.DEFINE_string('local_dir', '/tmp/mnist/wrong_images/', 
                    'where to save the wrong pictures, /tmp/mnist/wrong_images/ default.')

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

    best_accuracy = 0.0
    best_epoch = 0

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    y_conv = mnist_model.inference(x_image)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    for i in range(FLAGS.max_epoches):
        start_time = time.time()
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

        tra_loss = batch_average(mnist.train, cross_entropy, keep_dropout=1.0)
        tra_acc = batch_average(mnist.train, accuracy, keep_dropout=1.0)

        val_loss = batch_average(mnist.validation, cross_entropy, keep_dropout=1.0)
        val_acc = batch_average(mnist.validation, accuracy, keep_dropout=1.0)

        duration = time.time() - start_time
        print("epo %d:   tra_loss: %.4f   tra_acc: %.4f   val_loss: %.4f   val_acc: %.4f   dur: %.0f" 
              %(i, tra_loss, tra_acc, val_loss, val_acc, duration))

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = i
            saver.save(sess, 'my-model')
        if i - best_epoch > FLAGS.early_stop:
            print("Early Stopping")
            break
    


    print('training finished')

    for i in xrange(0, mnist.test.num_examples, FLAGS.batch_size):
        tmp_end = np.min([i+FLAGS.batch_size, mnist.test.num_examples])
        logits, temp = sess.run([y_conv, correct_prediction], feed_dict={
            x: mnist.test.images[i:tmp_end],
            y_: mnist.test.labels[i:tmp_end],
            keep_prob: 1.0})
        
        for k in xrange(len(temp)):
            if temp[k] == False:
                predicted_labels = np.argmax(logits[k], 0)
                actual_labels = np.argmax(mnist.test.labels[i+k], 0)
                wrong_images = np.ceil(mnist.test.images[i+k]*255).astype(np.uint8) 
                reshape_images = np.reshape(wrong_images, (28,28))
                # import pdb
                # pdb.set_trace()
                img = Image.fromarray(reshape_images)
                img.save(FLAGS.local_dir+'%d_%d_%d.png' 
                         %(predicted_labels, actual_labels, i+k))


    test_accuracy = batch_average(mnist.test, accuracy, keep_dropout=1.0)
    print("test accuracy %.5f" %test_accuracy)

def main(_):
    if tf.gfile.Exists(FLAGS.local_dir):
        tf.gfile.DeleteRecursively(FLAGS.local_dir)
    tf.gfile.MakeDirs(FLAGS.local_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
