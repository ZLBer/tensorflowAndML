import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import  numpy as np
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = to_categorical(train_df['target'])
X=train_df[features]
test=test_df[features]
Y=target

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.5);

ss = StandardScaler()
X=ss.fit_transform(X)
test=ss.fit_transform(test)


model=Sequential() # 模型初始化

model.add(Dense(input_dim=200,units=512,activation='relu')) # 添加一个全连接层,激活函数为relu
model.add(Dense(units=512,activation='relu'))  # 继续添加一层，只需要定义单元数目即可，输入由前一层决定
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='categorical_crossentropy', # 损失函数类型
              optimizer='adam',  # learnng rate更新方法
              metrics=['accuracy']) #定义性能评估方法，预测函数与目标函数的差异



# 添加训练数据, batch_size为每次训练数据额数目，epochs为迭代次数
model.fit(x_train,y_train,batch_size=1000,epochs=2)
score=model.evaluate(x_test,y_test)
print("Total loss on Testing Set:",score[0],"Accuracy of Testing Set:",score[1])

# predictions = model.predict_classes(test, verbose=0)
#
# submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
#                          "Label": predictions})
# submissions.to_csv("DR.csv", index=False, header=True)
id_code_test = test_df['ID_code']
print(model.predict_classes(x_test,verbose=1))
# Make predicitions
pred = model.predict_classes(test,verbose=0)
my_submission = pd.DataFrame({"ID_code" : id_code_test, "target" : pred})
my_submission.to_csv('submission.csv', index = False, header = True)