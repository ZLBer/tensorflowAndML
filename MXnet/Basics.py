from __future__ import print_function
import tensorflow as tf
import os

# # The default path for saving event files is the same folder of this python file.
# tf.app.flags.DEFINE_string(
# 'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
# 'Directory where event logs are written to.')
#
# # Store all elements in FLAG structure!
# FLAGS = tf.app.flags.FLAGS
#
# if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
#     raise ValueError('You must assign absolute path for --log_dir')

# Defining some constant values
a = tf.constant([5.0,6.0],name="a")
b = tf.constant(10.0,name="b")

x=tf.add(a,b,name="add")
y=tf.div(a,b,name="divide")

sess=tf.Session()
print(sess.run(x))

print(sess.run(y))

sess.close()
# # Run the session
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
#     print("output: ", sess.run([a,b,x,y]))
#
# # Closing the writer.
# writer.close()
# sess.close()