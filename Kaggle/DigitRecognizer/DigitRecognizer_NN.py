import pandas as pd
import  numpy as np
from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime

starttime = datetime.datetime.now()



df=pd.read_csv("train.csv")
X=np.array(df.iloc[:,1:],dtype='float32')
Y=to_categorical(df.label)
df=pd.read_csv("test.csv")
test=np.array(df)

X=X/255;
print(X[1])
print(Y[0])
print(X.shape)
print(Y.shape)
print(test.shape)


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2);

model=Sequential() # 模型初始化

model.add(Dense(input_dim=784,units=512,activation='relu')) # 添加一个全连接层,激活函数为relu
model.add(Dense(units=512,activation='relu'))  # 继续添加一层，只需要定义单元数目即可，输入由前一层决定
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy', # 损失函数类型
              optimizer='adam',  # learnng rate更新方法
              metrics=['accuracy']) #定义性能评估方法，预测函数与目标函数的差异



# 添加训练数据, batch_size为每次训练数据额数目，epochs为迭代次数
model.fit(x_train,y_train,batch_size=1000,epochs=22)
score=model.evaluate(x_test,y_test)
print("Total loss on Testing Set:",score[0],"Accuracy of Testing Set:",score[1])

predictions = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)


endtime = datetime.datetime.now()

print (endtime - starttime)