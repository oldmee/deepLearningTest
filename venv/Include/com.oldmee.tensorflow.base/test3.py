import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

new_value = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(new_value, feed_dict={input1:10.0,input2:15.0}))