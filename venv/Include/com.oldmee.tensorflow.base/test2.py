import tensorflow as tf

num = tf.Variable(0, name = "count")
new_value = tf.add(num, 10)
op = tf.assign(num, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(num))
    for i in range(5):
        sess.run(op)
        print(sess.run(num))