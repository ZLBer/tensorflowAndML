import tensorflow as tf

# 矩阵乘法
matrix1= tf.constant([[3,3]])
matrix2=tf.constant([
    [2],
    [2]
])

product = tf.matmul(matrix1,matrix2) # 矩阵乘法

# # 方法一
# sess= tf.Session()
# result = sess.run(product)
# print(result)
#
# sess.close()


with tf.Session as sess: # 打开session 并以sess命名，不用管关不关闭
   result2=sess.run(product)
   print(result2)