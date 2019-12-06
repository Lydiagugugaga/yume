# yume
Hello world
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#清除默认图形堆栈并重置全局默认图形，防止重复定义出错
tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
X_train,Y_train = mnist.train.images,mnist.train.labels  #导出训练集、测试集
X_test,Y_test = mnist.test.images,mnist.test.labels

#给X、Y定义成一个placeholder，即占位符

X = tf.placeholder(dtype=tf.float32,shape=[None,784],name='X')
Y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='Y')
#none为样本数，可变
# 定义各个参数，用tf自带的initializer
W1 = tf.get_variable('W1',[784,128],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2',[128,64],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3',[64,10],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3',[10],initializer=tf.zeros_initializer())

##激活函数

A1 = tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3 = tf.matmul(A2,W3)+b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

trainer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    # 首先给所有的变量都初始化（不用管什么意思，反正是一句必须的话）：
    sess.run(tf.global_variables_initializer())

    # 定义一个costs列表，来装迭代过程中的cost，从而好画图分析模型训练进展
    losses = []
    
    # 指定迭代次数：
    for it in range(10000):
        # 这里我们可以使用mnist自带的一个函数train.next_batch，可以方便地取出一个个地小数据集，从而可以加快我们的训练：
        X_batch,Y_batch = mnist.train.next_batch(batch_size=64)

# 我们最终需要的是trainer跑起来，并获得cost，所以我们run trainer和cost,同时要把X、Y给feed进去：
        _,batch_loss = sess.run([trainer,loss],feed_dict={X:X_batch,Y:Y_batch})
        losses.append(batch_loss)

        # 每100个迭代就打印一次cost：
        if it%100 == 0:
            print('iteration%d ,batch_loss: '%it,batch_loss)

    # 训练完成，我们来分别看看来训练集和测试集上的准确率：
    predictions = tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions,'float'))
    print("Training set accuracy: ",sess.run(accuracy,feed_dict={X:X_train,Y:Y_train}))
    print("Test set accuracy:",sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
