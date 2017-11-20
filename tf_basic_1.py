# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:01:24 2017

@author: rd0348
重新梳理一遍TensorFlow基础概念
"""
import tensorflow as tf

a=tf.zeros(shape=[1,2])

# 注意下面14行和15行输出结果区别
#with tf.Session() as sess:
#    print(sess.run(a));
#    print(a);
# 注意：根据上面的结论,在session.run之前，所有数据都是抽象的概念，
# 也就是说，a此时只是表示这应该是一个1*2的零矩阵，但却没有实际赋值
# 只有启动session后，才能得到a的值
# Session 会话
# Session就是抽象模型的实现者，具体的参数训练，预测，甚至是变量的实际值查询，都要用到session
# -----------------------------------------------------------------------------  
# Variable
# 当训练模型的时候，要用变量来存储和更新参数，变量包含张量（Tensor）存放于内存的缓存区，
# 建模时它们需要被明确地初始化，模型训练后它们b必须被存储到磁盘。
# 这些变量的值可在之后模型训练和分析时被加载。
# 如我要计算 y=ReLU(Wx+b)
# 那么W，b就是我要用来训练的参数，那么这两个值就可以用Variable来表示，
# Variable初始函数有很多选项，这里先不提，只输入一个tensor也是可以的
W = tf.Variable(tf.zeros((1,2)))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(W))
    
# 还可以用另一个变量的初始化值给当前变量初始化。
# 由于 tf.initialize_all_variables() 是并行地初始化所有变量，所以在有这种需求的情况下需要小心
weights = tf.Variable(tf.random_normal([784,200],stddev=0.35),name="weights")
w2 = tf.Variable(weights.initialized_value(),name="w2")
w_twice = tf.Variable(weights.initialized_value()*0.2,name="w_twice")

# -----------------------------------------------------------------------------
# placeholder 占位符
# 同样也是一个抽象的概念，用于表示输入输出数据的格式,
# 告诉系统：这里有一个值/向量/矩阵，现在我没发给你具体数值，
# 不过我正式运行的时候会补上，一般都是要从外部输入的值，
# 例如例子中的x和y,因为没有具体数值，只要指定尺寸就可以
# 需要注意: 如果在第二个参数赋了初始值,后面就不能够使用feed_dict赋值
x = tf.placeholder(tf.float32,name='input')
y = tf.placeholder(tf.float32,[1.,5],name='input')
with tf.Session() as sess:
    feed_dict = {x:[2.,7]} #这个必须是字典
    print(sess.run(x,feed_dict=feed_dict))
    #print(sess.run(y,feed_dict={y:[3.,9.]})) ' 这个赋值会导致编译报错,因为y 已经赋了[None,5]

