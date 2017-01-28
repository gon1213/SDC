import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
from sklearn.utils import shuffle

saver = tf.train.Saver()

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
	data = pickle.load(f)
x_data = data["features"]
y_data = data["labels"]
# TODO: Split data into training and validation sets.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=0)
# TODO: Define placeholders and resize operation.
sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
epochs = 10
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)

fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = (tf.Variable(tf.zeros(nb_classes)))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)



# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
init_op = tf.global_variables_initializer()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

# TODO: Train and evaluate the feature extraction model.

def evaluate(x_data, y_data, sess):
    num_examples = x_data.shape[0]

    total_loss = 0
    total_accuracy = 0

    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):

        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

        loss, accuracy = sess.run([loss_operation,accuracy_op], feed_dict={x: batch_x, y: batch_y})

        total_accuracy += (accuracy * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_accuracy / num_examples

import time
start = time.clock()
with tf.Session() as sess:
    sess.run(init_op)
    num_examples = len(x_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_t, y_t = shuffle(x_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_t[offset:end], y_t[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_loss, validation_accuracy = evaluate(x_test, y_test, sess)
        print("EPOCH {} ...".format(i+1))
        print("validation_loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Time use:  {:2f}s".format(time.time()-t0))
        
    saver.save(sess, './Alexnet_traffic_sign')
    print("Model saved")
end = time.clock()
print ("Total time used:  {:2f}s".format(end - start))


########################################################################################
#Training...

# EPOCH 1 ...
# validation_loss = 0.524
# Validation Accuracy = 0.865
# Time use:  2649.567318s
# EPOCH 2 ...
# validation_loss = 0.340
# Validation Accuracy = 0.916
# Time use:  2612.787713s
# EPOCH 3 ...
# validation_loss = 0.269
# Validation Accuracy = 0.932
# Time use:  2618.165683s
# EPOCH 4 ...
# validation_loss = 0.222
# Validation Accuracy = 0.945
# Time use:  2679.099313s
# EPOCH 5 ...
# validation_loss = 0.192
# Validation Accuracy = 0.951
# Time use:  2846.550134s
# EPOCH 6 ...
# validation_loss = 0.167
# Validation Accuracy = 0.960
# Time use:  2594.750446s
# EPOCH 7 ...
# validation_loss = 0.157
# Validation Accuracy = 0.959
# Time use:  2604.869899s
# EPOCH 8 ...
# validation_loss = 0.147
# Validation Accuracy = 0.961
# Time use:  2603.650970s
# EPOCH 9 ...
# validation_loss = 0.137
# Validation Accuracy = 0.963
# Time use:  2606.993917s
# EPOCH 10 ...
# validation_loss = 0.129
# Validation Accuracy = 0.964
# Time use:  2517.727100s
# Total time used:  85544.036140s



