############################################################
# Original from:                                           #
# http://tensorflow.org/tutorials/mnist/beginners/index.md #
# Copyright 2015 TensorFlow Authors.  Apache Version 2.0   #
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy
import timeit

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


############################################################


target_error = 0.95
mbatchsz     = 100
maxiters     = 2000
FLAGS        = None


############################################################


def main(_):
  ##############################

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # Set NN topology
  topology = [FLAGS.c1_sz, FLAGS.c1_ft, FLAGS.c2_sz, FLAGS.c2_ft, FLAGS.fullyc_sz]
  print("NN Topology: ", topology)

  ##############################

  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  W_conv1 = weight_variable([topology[0], topology[0], 1, topology[1]])
  b_conv1 = bias_variable([topology[1]])

  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([topology[2], topology[2], topology[1], topology[3]])
  b_conv2 = bias_variable([topology[3]])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * topology[3], topology[4]])
  b_fc1 = bias_variable([topology[4]])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*topology[3]])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([topology[4], 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(FLAGS.momentum).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.global_variables_initializer())

  elapsed = 0.0
  iters   = maxiters
  for i in range(iters):
    batch = mnist.train.next_batch(mbatchsz)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      val_acc = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
      if i == 0:
        print("#iter traintime train-err val-err")
      print("%d %e %e %e"%(i,elapsed,train_accuracy,val_acc))
      if val_acc >= target_error:
        if iters == i-100:
          iters=i
          break
        iters=i
    start_time = timeit.default_timer()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout})
    if i >= 100:
      elapsed += timeit.default_timer() - start_time

  ##############################

  test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  print("\nTime to error (%f): %e s"%(1.0-target_error,elapsed))
  print("Train iters:          %d"%(iters))
  print("Test accuracy:        %e\n"%test_acc)
  print("FoM: %e"%(elapsed))

  ##############################


############################################################


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory for storing input data')
  # Topology
  parser.add_argument('--c1_sz', type=int, default=5)
  parser.add_argument('--c1_ft', type=int, default=32)
  parser.add_argument('--c2_sz', type=int, default=5)
  parser.add_argument('--c2_ft', type=int, default=64)
  parser.add_argument('--fullyc_sz', type=int, default=1024)

  parser.add_argument('--dropout', type=float, default='0.5',
                      help='Probability of dropout.')
  parser.add_argument('--momentum', type=float, default='1e-4',
                      help='Momentum for AdamOptimizer.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


############################################################
