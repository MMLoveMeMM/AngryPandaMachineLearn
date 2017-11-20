# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:00:47 2017

@author: rd0348
两层卷积网络实现训练MNIST数据集
"""

from tensorflow.examples.tutorials.mnist import input_data  
  
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)  

import tensorflow as tf

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

def weight_variable(shape):
    initial =tf.truncated_normal(shape,stddev=0.1) # 变量的初始化值为截断正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape) #变量初始化设为常数,不为0的小数
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME') # 此处要求卷积后图像大小不变，padding的值为SAME

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#sess.run(tf.global_variables_initializer())
sess.close()   

# 第一层卷积
# 第一层卷积核（filter）的尺寸是5*5，通道数为1（因为原图通道为1），输出通道为32（32个滤波器），即feature map数目为32
# 又因为strides=[1,1,1,1]所以单个通道的输出尺寸应该跟输入图像一样，即总的卷积输出为？*28*28*32，？是批次数
# 在池化阶段，ksize=[1,2,2,1]那么卷积的结果经过池化后，其尺寸应该是？*14*14*32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
#卷积核5*5，输入通道为32，输出通道为64
#卷积前图像的尺寸为？*14*14*32，卷积后为？*14*14*64
#池化后，输出的图像尺寸为？*7*7*64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 输入维度为7*7*64，输出维度为1024
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# 这里使用了Dropout技术，随机安排一些cell输出值为0，防止过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# 分类层
# 输入1024维，输出10维，也就是具体的0~9分类
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 定义损失函数，优化方法
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables()) # 变量初始化  

#sess = tf.Session()
#sess.run(tf.initialize_all_variables())
#sess.run(tf.global_variables_initializer())
#sess.close()   

for i in range(20000):  
    batch = mnist.train.next_batch(50)  
    if i%100 == 0:  
        # print(batch[1].shape)  
        train_accuracy = accuracy.eval(feed_dict={  
            x:batch[0], y_: batch[1], keep_prob: 1.0})  
        print("step %d, training accuracy %g"%(i, train_accuracy))  
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  
  
print("test accuracy %g"%accuracy.eval(feed_dict={  
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) #注意测试集测试时不使用dropout

"""
先介绍一下所用核心函数
1.tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,data_format=None,name=None)
input：待卷积的数据，格式要求为一个张量tensor，【batch,in_height,in_width,in_channels】
分别表示 批次数，图像高度，宽度，输入通道数
filter：卷积核，格式要求为【filter_height,filter_width,in_channels,out_channels】
分别表示 卷积核的高度，宽度，输入通道数，输出通道数

strides:一个长为4的list，表示每次卷积以后卷积窗口在input中滑动的距离
padding:有SAME和VALID两个选项，表示是否要保留图像边上那一圈不完全卷积的部分，如果是SAME，则保留
use_cudnn_on_gpu:是否使用cudnn加速，默认是True

2.tf.nn.max_pool(value,ksize,strides,padding,data_format="NHWC",name=None)
value: 一个4维的张量，格式为【batch,height,width,channels】与conv2d中input格式一样
ksize:长为4的list，表示池化窗口的尺寸
strides:池化窗口的滑动值，与conv2d中的一样
padding:与conv2d中用法一样
"""




