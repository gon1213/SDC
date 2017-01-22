# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
x = tf.constant(10)
y = tf.constant(2)
a = tf.constant(1)
z = tf.sub(tf.div(x,y),a)
with tf.Session() as sess:
    output = sess.run(z)
    print(output)

# like python variable, it lets tensonflow can change the number
x = tf.Variable(5)

#initialize all the variable
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)

# The tf.truncated_normal() function returns a tensor with random values from a normal distribution 
# whose magnitude is no more than 2 standard deviations from the mean.
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))


#tf.zeros return a tensor with all zeros
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

softmax = tf.nn.softmax(logits)

x = tf.reduce_sum([1,2,3,4,5]) #15

x = tf.log(100)

