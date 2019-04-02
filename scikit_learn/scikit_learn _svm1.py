from sklearn import svm
#训练样本
x = [[2,0], [1,1], [2,3]]
#label
y = [0,0,1]

clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

#打印出参数设置情况,只设置了 kernel，其他都是默认
print(clf)

#支持向量
print(clf.support_vectors_)

#支持向量的index
print(clf.support_)

#对于每个类别，分别有几个支持向量
print(clf.n_support_)

#对新数据进行预测
print(clf.predict([[2,0]]))
