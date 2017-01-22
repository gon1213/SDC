import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)


with tf.Session() as sess:
	output = sess.run(x, feed_dict={x:"Hello World"})
	print(output)


with tf.Session() as sess:
	output = sess.run(x , feed_dict={x:'Test String', y: 123, z: 45.67})
	print(output)

a = tf.add(5, 2)

b = tf.sub(10, 4)

c = tf.nul(2 ,5)
