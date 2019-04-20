import pandas as pd
import  numpy as np
from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("train.csv")
X=np.array(df.iloc[:,1:],dtype='float32')
Y=to_categorical(df.label)
df=pd.read_csv("test.csv")
test=np.array(df)

X=X/255;

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2);

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
# 太慢了点