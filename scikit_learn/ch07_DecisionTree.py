import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

#
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print('train score: {0}; test score: {1}'.format(train_score, test_score))
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)


#用网格搜索进行调参
entropy_thresholds = np.linspace(0, 0.01, 50)
gini_thresholds = np.linspace(0, 0.005, 50)
param_grid = [{'criterion': ['entropy'],
               'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'],
               'min_impurity_decrease': gini_thresholds}, #  min_impurity_decrease 节点划分最小不纯度 ,防止过拟合
              {'max_depth': range(2, 10)},
              {'min_samples_split': range(2, 30, 2)}] #每个字典都要穷举遍历
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, return_train_score=True) # cv交叉验证的折数
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,
                                                clf.best_score_))



# 根据找到的最佳参数生成决策树
clf = DecisionTreeClassifier(criterion=clf.best_params_['criterion'], min_impurity_decrease=clf.best_params_['min_impurity_decrease'])
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))