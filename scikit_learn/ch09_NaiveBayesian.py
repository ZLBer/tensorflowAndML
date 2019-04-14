import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#特征工程
def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理性别数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理登船港口数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data
train = read_dataset('datasets/titanic/train.csv')
print(train.head())
y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values


# 生成交叉验证集合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('train dataset: {0}; test dataset: {1}'.format(
    X_train.shape, X_test.shape))



#mnb = MultinomialNB()   # train score: 0.6853932584269663; test score: 0.664804469273743
mnb=GaussianNB() # train score: 0.7837078651685393; test score: 0.7932960893854749 ,发现效果并不是很好
mnb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
train_score = mnb.score(X_train, y_train)
test_score = mnb.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))