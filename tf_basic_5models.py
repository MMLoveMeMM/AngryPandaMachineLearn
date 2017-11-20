# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:46:47 2017

@author: rd0348
"""

from tensorflow.examples.tutorials.mnist import input_data  
  
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)  
# 下面的已经下载好的,如果没有穿墙到google
#mnist = input_data.read_data_sets("C:\\Users\\1\\AppData\\Local\\Programs\\Python\Python35\\Lib\\site-packages\\tensorflow\\examples\\tutorials\\mnist", one_hot=True)  
import tensorflow as tf
# 定义每层的神经元个数
K=400
L=100
M=60
N=30

# 搭建模型
W1=tf.Variable(tf.truncated_normal([28*28,K],stddev=0.1))  
B1=tf.Variable(tf.zeros([K]))  
W2=tf.Variable(tf.truncated_normal([K,L],stddev=0.1))  
B2=tf.Variable(tf.zeros([L]))  
W3=tf.Variable(tf.truncated_normal([L,M],stddev=0.1))  
B3=tf.Variable(tf.zeros([M]))  
W4=tf.Variable(tf.truncated_normal([M,N],stddev=0.1))  
B4=tf.Variable(tf.zeros([N]))  
W5=tf.Variable(tf.truncated_normal([N,10],stddev=0.1))  
B5=tf.Variable(tf.zeros([10]))  
  
X=tf.placeholder(tf.float32,[None,28,28,1])  
y_=tf.placeholder(tf.float32,[None,10]) #hot-vector的形式  
X=tf.reshape(X,[-1,28*28])  
Y1=tf.nn.relu(tf.matmul(X,W1)+B1)  
Y2=tf.nn.relu(tf.matmul(Y1,W2)+B2)  
Y3=tf.nn.relu(tf.matmul(Y2,W3)+B3)  
Y4=tf.nn.relu(tf.matmul(Y3,W4)+B4)  
pred=tf.nn.softmax(tf.matmul(Y4,W5)+B5)  
  
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y_))  
train_step=tf.train.GradientDescentOptimizer(0.03).minimize(loss)  
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y_,1)) #找到每行最大的作为输出结果  
accu=tf.reduce_mean(tf.cast(corr,tf.float32))  
init=tf.global_variables_initializer() 

# 开始训练
sess=tf.InteractiveSession()  
sess.run(init)  
for i in range(100000):  
    batch_xs, batch_ys = mnist.train.next_batch(100)  
    sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})  
    if i % 100 ==0:  
        print("测试集正确率：%f" %accu.eval(feed_dict={X:mnist.test.images,y_:mnist.test.labels},session=sess))  


