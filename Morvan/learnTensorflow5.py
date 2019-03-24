import  tensorflow as tf
import  numpy  as np
import  matplotlib.pyplot as plt
import os
def add_layer(intputs,in_size,out_size,activation_function=None):
    # tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))+0.1
    print("inputs",intputs)
    print("weight",Weights)
    print("biases",biases)
    Wx_plus_b=tf.matmul(intputs,Weights)+biases
    print("Wx_pul",Wx_plus_b)
    if activation_function  is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

# 等差数列函数
x_data=np.linspace(-1,1,300)[:,np.newaxis] # 将行转成列

noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise


xs = tf.placeholder(tf.float32,[None,1],name="x")  # [None, 1]表示列是1，行不定
ys  =tf.placeholder(tf.float32,[None,1],name="y")

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

predition=add_layer(l1,10,1,activation_function=None)
# reduction_indices，直译过来就是指“坍塌维度”，即按照哪个维度进行加法运算。ba'hang
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]),name="LOSS")
# [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
# [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)


# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
'Directory where event logs are written to.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')
writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50== 0:
       print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
       try:
           ax.lines.remove(lines[0])
       except Exception:
           pass
       predition_value= sess.run(predition,feed_dict={xs:x_data,ys:y_data})
       lines= ax.plot(x_data,predition_value,'r-',lw=5)
       plt.pause(0.1)