import  pandas as  pd
import numpy as np
from keras import Sequential
from keras.layers import Dense



# 数据预处理
def Traindata_handing(path):
    df = pd.read_csv(path)
     # 空值处理
    age_mean = df['Age'].mean()
    fare_mean = df['Fare'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    df['Fare'] = df['Fare'].fillna(fare_mean)
    df['Embarked'] = df['Embarked'].fillna('S')
    # 数据替换
    df.Embarked[df['Embarked'] == 'S'] = 0
    df.Embarked[df['Embarked'] == 'C'] = 1
    df.Embarked[df['Embarked'] == 'Q'] = 2
    # df.Sex[df['Sex'] == 'male'] = 1
    # df.Sex[df['Sex'] == 'female'] = 0

    df['Sex']=df['Sex'].map({'male':0,'female':1}).astype(int)
    x_Train = df[[ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]
    y_Train=pd.get_dummies(df.Survived)
    x_Train = np.array(x_Train)
    y_Train = np.array(y_Train)
    print(y_Train)
    return x_Train,y_Train

def Predictdata_handing(path):
    df = pd.read_csv(path)
    df.Sex[df['Sex'] == 'male'] = 1
    df.Sex[df['Sex'] == 'female'] = 0
    # print(df.isnull().sum())
    age_mean = df['Age'].mean()
    fare_mean = df['Fare'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    df['Fare'] = df['Fare'].fillna(fare_mean)
    df['Embarked'] = df['Embarked'].fillna('S')
    df.Embarked[df['Embarked'] == 'S'] = 0
    df.Embarked[df['Embarked'] == 'C'] = 1
    df.Embarked[df['Embarked'] == 'Q'] = 2
    x_predict2 = df[[ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    x_predict = np.array(x_predict2)
    return x_predict





x_Train,y_Train=Traindata_handing('train.csv')

x_predict=Predictdata_handing('test.csv')


df = pd.read_csv('test.csv')
x_predict2=df[['PassengerId','Sex','Age','SibSp','Parch','Fare','Embarked']]

#
#
model=Sequential() # 模型初始化
model.add(Dense(input_dim=x_Train.shape[1],units=100,activation='relu')) # 添加一个全连接层,激活函数为relu
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='categorical_crossentropy', # 损失函数类型
              optimizer='Adam',  # learnng rate更新方法
              metrics=['accuracy']) #定义性能评估方法，预测函数与目标函数的差异

# 添加训练数据, batch_size为每次训练数据额数目，epochs为迭代次数
model.fit(x_Train,y_Train,batch_size=30,epochs=10)


# score=model.evaluate(x_test,y_test)
# print("Total loss on Testing Set:",score[0],"Accuracy of Testing Set:",score[1])

list=[]
predictions=model.predict_classes(x_predict)
print(predictions)

#
# # print(type(x_predict2['PassengerId']))
# # print(type(predictions))
# #
result = pd.DataFrame({'PassengerId':x_predict2['PassengerId'], 'Survived':predictions})
result.to_csv("logistic_regression_predictions.csv", index=False)


