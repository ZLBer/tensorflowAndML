import tensorflow as tf

# placeholder 相当于占位符，后期传入数值


#input1=tf.placeholder(tf.float32,[2,2])
input1= tf.placeholder(tf.float32)
input2= tf.placeholder(tf.float32)


output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))