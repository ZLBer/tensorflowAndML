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
print("y",y.shape)

w=np.random.normal(x[0])
w=np.reshape(w,(-1,1))
print("w",w.shape)
bias=np.zeros(1)

# y_predict=np.matmul(x,w)+bias;
# print(y_predict)
# print("y_predict",y_predict.shape)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)      #在feature基础上加入bias
# 初始化一个参数矩阵
w=np.zeros((len(x[0])))

#初始化一个learning rate
lr=0.01
iteration=10000  #迭代10000次
s_grad=np.zeros(len(x[0]))

print("x",x.shape)
print("w",w)
print(np.matmul(x,w))
for i in range(iteration):
    tem=np.dot(x,w)     #&y^*&(预测值)
    loss=y-tem
    if i%50==0:
     print(np.mean(loss))
    grad=np.dot(x.transpose(),loss)*(-2) #该次的梯度
    s_grad+=grad**2  # 梯度的和
    ada=np.sqrt(s_grad)
    w=w-lr*grad/ada # 重新求w
print(w)



# ------------------测试数据
testdata=pd.read_csv('test.csv')
print(testdata)
pm2_5_test=testdata[testdata['AMB_TEMP']=='PM2.5'].ix[:,2:]
x_test=np.array(pm2_5_test,float)
print(x_test)
x_test_b=np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)
y_star=np.dot(x_test_b,w)


real=pd.read_csv('ans.csv')
erro=abs(y_star-real.value).sum()/len(real.value)