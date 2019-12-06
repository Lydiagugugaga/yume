# yume
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 清除默认图形堆栈并重置全局默认图形，防止重复定义出错
tf.reset_default_graph()

# 导入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
# 导出训练集、测试集
X_train,Y_train = mnist.train.images,mnist.train.labels 
X_test,Y_test = mnist.test.images,mnist.test.labels

# 给X、Y定义成一个placeholder，即占位符
X = tf.placeholder(dtype=tf.float32,shape=[None,784],name='X')
Y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='Y')   #none为样本数，可变

# 定义各个参数，用tensorflow框架自带的initializer进行参数初始化
# 在tf.contrib.layers高级函数模块里进行初始化权重，w不能为0，b可以为0
W1 = tf.get_variable('W1',[784,128],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2',[128,64],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3',[64,10],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3',[10],initializer=tf.zeros_initializer())

# 在tf.nn模块中调用relu激活函数
A1 = tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3 = tf.matmul(A2,W3)+b3

# 计算损失函数，softmax-loss模型
# reduce_mean()函数计算平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

# 采用Adam优化器，minimize计算损失函数梯度，学习率设为0.01
trainer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 启动搭建好的计算图
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    # 定义一个losses列表，来装迭代过程中的loss
    losses = []
    
    # 指定迭代次数为1000
    for it in range(1000):
        # 通过mnist自带函数train.next_batch，制作小批量数据集，提升计算效率：
        X_batch,Y_batch = mnist.train.next_batch(batch_size=64)

    # 利用feed_dict函数X、Y赋值：
        _, batch_loss = sess.run([trainer,loss],feed_dict={X:X_batch,Y:Y_batch})
        losses.append(batch_loss)

        # 每100个迭代就打印一次损失：
        if it%100 == 0:
            print('iteration%d ,batch_loss: '%it,batch_loss)

    # 计算训练集和测试集上的准确率，equal函数判断是否相等，argmax返回,transpose转置
    predictions = tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions,'float'))
    print("Training set accuracy: ",sess.run(accuracy,feed_dict={X:X_train,Y:Y_train}))
    print("Test set accuracy:",sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
