import numpy as np
import pandas as pd
import  tensorflow as tf

TrainData = pd.read_csv('train.csv', encoding="UTF-8")

pm2_5=TrainData[TrainData['測項']=='PM2.5'].ix[:,3:]

tempxlist=[]
tempylist=[]
for i in range(15):
    tempx=pm2_5.iloc[:,i:i+9]        #使用前9小时数据作为feature
    tempx.columns=np.array(range(9))
    tempy=pm2_5.iloc[:,i+9]         #使用第10个小数数据作为lable
    tempy.columns=['1']
    tempxlist.append(tempx)
    tempylist.append(tempy)

xdata=pd.concat(tempxlist)     #组合成一组feature数据
x=np.array(xdata,dtype='float32')  # 构造np数组
print("x",x.shape)


ydata=pd.concat(tempylist)      #lable数据
ydata.head()
y=(np.array(ydata,float))
y=np.reshape(y,(-1,1))
print("y",y.shape)

w=np.random.normal(x[0])
w=np.reshape(w,(-1,1))
print("w",w.shape)
bias=np.zeros(1)

y_predict=np.matmul(x,w)+bias;
print(y_predict)
print("y_predict",y_predict.shape)

#
# lr=0.000001
# literation=99999
# loss=(y-y_predict)**2
# #  2*x*(y-y_predict)
#
# print((y-y_predict).shape)
#
# #更新w
# for i in range(literation):
#   y_predict = np.matmul(x, w) + bias;
#   w=w-np.transpose(lr*2*np.matmul(np.transpose((y-y_predict)),x)/len(y))
#   loss = (y - y_predict) ** 2
#   print("www",w)
#   print(np.mean(loss))

#----------------------------tensorflow实现
x_data = x
y_data = y

# 开始创建结构

Weights = tf.Variable(tf.random_uniform([9,1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y=tf.matmul(x_data,Weights)+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.AdagradOptimizer(0.1)
train=optimizer.minimize(loss)

init =tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)  # 激活

for step in range(999999):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
        print(sess.run(loss))