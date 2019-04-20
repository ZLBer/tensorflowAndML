import pandas as pd
import  numpy as np
from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime





df=pd.read_csv("train.csv")
X=np.array(df.iloc[:,1:].values.reshape(-1,28,28,1),dtype='float32')
Y=to_categorical(df.label)
#df=pd.read_csv("test.csv")
#test=np.array(df)

X=X/255;

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1);

model=Sequential() # 模型初始化


#convolutional (Conv2D) layer
#即定义卷积层，32个过滤器=输出层的厚度
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#The Flatten layer is use to convert the final feature maps into a one single 1D vector.
# 就是说把多维矩阵转成一维向量
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=86,epochs=1)
score=model.evaluate(x_test,y_test)

print("score",score)