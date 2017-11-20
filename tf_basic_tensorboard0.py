# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:08:06 2017

@author: rd0348
"""

import tensorflow as tf

LOCAL_TENSORFLOW_DATA = "mnist_tensorboard0\\"

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
sess.run(e)

writer = tf.summary.FileWriter(LOCAL_TENSORFLOW_DATA, tf.get_default_graph())
writer.close()
sess.close()