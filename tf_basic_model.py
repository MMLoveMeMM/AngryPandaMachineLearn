# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:06:05 2017

@author: rd0348
http://blog.csdn.net/margretwg/article/details/70158930
这个线性回归讲的到位
http://blog.csdn.net/xbinworld/article/details/43919445
"""
from tensorflow.examples.tutorials.mnist import input_data  
  
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)  
# 下面的已经下载好的,如果没有穿墙到google
#mnist = input_data.read_data_sets("C:\\Users\\1\\AppData\\Local\\Programs\\Python\Python35\\Lib\\site-packages\\tensorflow\\examples\\tutorials\\mnist", one_hot=True)  
import tensorflow as tf

# 对输入x，label y_创建一个占位符，以及声明W,b变量,通过softmax得到预测结果y
# 定义损失函数cross_entropy，训练方法（梯度下降）train_step
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 建立抽象模型
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()

# 实际训练
# 在模型搭建完成以后，我们只要为模型提供输入和输出，模型就能够自己进行训练和测试，
# 中间的求导，求梯度，反向传播等等，TensorFlow都会帮你自动完成。结果最终为0.92
with tf.Session() as sess:
    sess = tf.InteractiveSession()
    sess.run(init)
    for i in range(10000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if i%500==0:
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}, session=sess))