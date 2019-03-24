import numpy as np
import pandas as pd
# TrainData = pd.read_csv('train.csv', encoding="UTF-8")
# # 从头取
# print(TrainData.head(5))
# # 从尾部取
# print(TrainData.tail(5))
# print (TrainData.shape)
#
# pd.options.display.float_format = '{:,.3f}'.format
#
# print(TrainData.describe())
#
#
# # Pandas将返回一个series，而不是一个dataframe  可以将dataframe视作series的字典。所以，如果我们取出了某一列，我们获得的自然是一个series。
# # print(TrainData['日期'])
# # print(TrainData.日期) #功能类似
#
#
# #条件过滤
#
# print(TrainData[TrainData.日期=='2014/1/1'])
# TrainData[TrainData.日期.str.startswith('2014')]
#
#
#
# # loc是字符串标签的索引方法，iloc是数字标签的索引方法
# #选出某行,数字索引
# print(TrainData.iloc[30])
# # 选出某行，字符串索引
# #TrainData.loc['12']
#
# # 调用sort_index来对dataframe实现排序
# #TrainData.sort_index(ascending=False)
#
# print(TrainData[TrainData['測項']=='PM2.5'])


l1=np.array([
    [1,2],
    [1,2]
])
l2=[1,2]
# 好厉害的函数..
print(np.matmul(l2,l1))