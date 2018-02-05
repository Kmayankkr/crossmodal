
import tensorflow as tf
import numpy as np
from scipy import ndimage, misc
import nus_wide_10k_loader
import pdb, sys, os

def get_conv_layer(input_data, conv_filt_shape, name):
    # weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(shape=conv_filt_shape, stddev=0.03, dtype=tf.float32), name=name+'_weights')
    bias = tf.Variable(tf.truncated_normal([conv_filt_shape[3]], stddev=0.03, dtype=tf.float32), name=name+'_bias')

    # convolutional layer operation
    output_data = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # bias
    output_data += bias

    # ReLU non-linear activation
    output_data = tf.nn.relu(output_data)

    return output_data

def get_pool_layer(input_data, pool_shape, pool_strides, name):
    # max pooling
    ksize = [1] + pool_shape + [1]
    output_data = tf.nn.max_pool(input_data, ksize=ksize, strides=pool_strides, padding='SAME', name=name+'_pool')

    return output_data

def get_weight(shape, name):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.03, dtype=tf.float32), name=name+'_weights')
    return weights

def get_bias(shape, name):
    bias = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.03, dtype=tf.float32), name=name+'_bias')
    return bias

def get_dense_layer(input_data, num_nodes, activation, name):
    input_data_shape = input_data.get_shape().as_list()

    # weights and bias for the filter
    weights = get_weight([input_data_shape[1], num_nodes], name+'_weights')
    bias = get_bias([input_data_shape[0], num_nodes], name+'_bias')

    # fully connected layer operation
    output_data = activation(tf.matmul(input_data, weights) + bias, name=name)

    return output_data

def get_dropout_layer(input_data, keep_prob, name):
    # dropout
    output_data = tf.nn.dropout(input_data, keep_prob=keep_prob, name=name)

    return output_data

batch_size = 20

source_image_input = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], 'source_image')
source_text_input = tf.placeholder(tf.float32, [batch_size, 300], 'source_text')
source_label_input = tf.placeholder(tf.float32, [batch_size, 10], 'source_label')

target_image_input = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], 'target_image')
target_text_input = tf.placeholder(tf.float32, [batch_size, 300], 'target_text')
target_label_input = tf.placeholder(tf.float32, [batch_size, 10], 'target_label')

# architecture

# source image

with tf.variable_scope("source_image"):
    filter1_shape = [5, 5, 3, 64]
    SI_conv1 = get_conv_layer(source_image_input, filter1_shape, 'SI_conv1')

    pool1_shape = [3, 3]
    pool1_strides = [1, 2, 2, 1]
    SI_pool1 = get_pool_layer(SI_conv1, pool1_shape, pool1_strides, 'SI_pool1')

    filter2_shape = [3, 3, filter1_shape[3], 32]
    SI_conv2 = get_conv_layer(SI_pool1, filter2_shape, 'SI_conv2')

    pool2_shape = [3, 3]
    pool2_strides = [1, 2, 2, 1]
    SI_pool2 = get_pool_layer(SI_conv2, pool2_shape, pool2_strides, 'SI_pool2')

    filter3_shape = [3, 3, filter2_shape[3], 16]
    SI_conv3 = get_conv_layer(SI_pool2, filter3_shape, 'SI_conv3')

    pool3_shape = [3, 3]
    pool3_strides = [1, 2, 2, 1]
    SI_pool3 = get_pool_layer(SI_conv3, pool3_shape, pool3_strides, 'SI_pool3')

    filter4_shape = [3, 3, filter3_shape[3], 8]
    SI_conv4 = get_conv_layer(SI_pool3, filter4_shape, 'SI_conv4')

    pool4_shape = [3, 3]
    pool4_strides = [1, 2, 2, 1]
    SI_pool4 = get_pool_layer(SI_conv4, pool4_shape, pool4_strides, 'SI_pool4')

    SI_dense1 = tf.reshape(SI_pool4, [batch_size, -1], 'SI_dense1')

    SI_hidden = SI_dense1

# source text

with tf.variable_scope("source_text"):
    text_shape = source_text_input.get_shape().as_list()

    ST_dense1_num = 1024
    ST_dense1 = get_dense_layer(source_text_input, ST_dense1_num, tf.nn.relu, 'ST_dense1')
    ST_dropout1 = get_dropout_layer(ST_dense1, 1, 'ST_dropout1')

    ST_dense2_num = 2048
    ST_dense2 = get_dense_layer(ST_dropout1, ST_dense2_num, tf.nn.relu, 'ST_dense2')
    ST_dropout2 = get_dropout_layer(ST_dense2, 1, 'ST_dropout2')

    ST_hidden = ST_dropout2

# target image

with tf.variable_scope("target_image"):
    filter1_shape = [5, 5, 3, 64]
    TI_conv1 = get_conv_layer(target_image_input, filter1_shape, 'TI_conv1')

    pool1_shape = [3, 3]
    pool1_strides = [1, 2, 2, 1]
    TI_pool1 = get_pool_layer(TI_conv1, pool1_shape, pool1_strides, 'TI_pool1')

    filter2_shape = [3, 3, filter1_shape[3], 32]
    TI_conv2 = get_conv_layer(TI_pool1, filter2_shape, 'TI_conv2')

    pool2_shape = [3, 3]
    pool2_strides = [1, 2, 2, 1]
    TI_pool2 = get_pool_layer(TI_conv2, pool2_shape, pool2_strides, 'TI_pool2')

    filter3_shape = [3, 3, filter2_shape[3], 16]
    TI_conv3 = get_conv_layer(TI_pool2, filter3_shape, 'TI_conv3')

    pool3_shape = [3, 3]
    pool3_strides = [1, 2, 2, 1]
    TI_pool3 = get_pool_layer(TI_conv3, pool3_shape, pool3_strides, 'TI_pool3')

    filter4_shape = [3, 3, filter3_shape[3], 8]
    TI_conv4 = get_conv_layer(TI_pool3, filter4_shape, 'TI_conv4')

    pool4_shape = [3, 3]
    pool4_strides = [1, 2, 2, 1]
    TI_pool4 = get_pool_layer(TI_conv4, pool4_shape, pool4_strides, 'TI_pool4')

    TI_dense1 = tf.reshape(TI_pool4, [batch_size, -1], 'TI_dense1')

    TI_hidden = TI_dense1

# target text

with tf.variable_scope("target_text"):
    text_shape = target_text_input.get_shape().as_list()

    TT_dense1_num = 1024
    TT_dense1 = get_dense_layer(target_text_input, TT_dense1_num, tf.nn.relu, 'TT_dense1')
    TT_dropout1 = get_dropout_layer(TT_dense1, 1, 'TT_dropout1')

    TT_dense2_num = 2048
    TT_dense2 = get_dense_layer(TT_dropout1, TT_dense2_num, tf.nn.relu, 'TT_dense2')
    TT_dropout2 = get_dropout_layer(TT_dense2, 1, 'TT_dropout2')

    TT_hidden = TT_dropout2

# classification

SI_hidden_shape = SI_hidden.get_shape().as_list()
ST_hidden_shape = ST_hidden.get_shape().as_list()
TI_hidden_shape = TI_hidden.get_shape().as_list()
TT_hidden_shape = TT_hidden.get_shape().as_list()

with tf.variable_scope("common_layer_1"):
    C_dense1_num = 512
    C_dense1_weight = get_weight([SI_hidden_shape[1], C_dense1_num], 'C_dense1_weight')
    C_dense1_bias = get_bias([SI_hidden_shape[0], C_dense1_num], 'C_dense1_bias')

    CSI_dense1 = tf.nn.relu(tf.matmul(SI_hidden, C_dense1_weight) + C_dense1_bias, name='CSI_dense1')
    CSI_dropout1 = get_dropout_layer(CSI_dense1, 1, 'CSI_dropout1')

    CST_dense1 = tf.nn.relu(tf.matmul(ST_hidden, C_dense1_weight) + C_dense1_bias, name='CST_dense1')
    CST_dropout1 = get_dropout_layer(CST_dense1, 1, 'CST_dropout1')

    CTI_dense1 = tf.nn.relu(tf.matmul(TI_hidden, C_dense1_weight) + C_dense1_bias, name='CTI_dense1')
    CTI_dropout1 = get_dropout_layer(CTI_dense1, 1, 'CTI_dropout1')

    CTT_dense1 = tf.nn.relu(tf.matmul(TT_hidden, C_dense1_weight) + C_dense1_bias, name='CTT_dense1')
    CTT_dropout1 = get_dropout_layer(CTT_dense1, 1, 'CTT_dropout1')

with tf.variable_scope("common_layer_2"):
    C_dense2_num = 128
    C_dense2_weight = get_weight([C_dense1_num, C_dense2_num], 'C_dense2_weight')
    C_dense2_bias = get_bias([SI_hidden_shape[0], C_dense2_num], 'C_dense2_bias')

    CSI_dense2 = tf.nn.relu(tf.matmul(CSI_dropout1, C_dense2_weight) + C_dense2_bias, name='CSI_dense2')
    CSI_dropout2 = get_dropout_layer(CSI_dense2, 1, 'CSI_dropout2')

    CST_dense2 = tf.nn.relu(tf.matmul(CST_dropout1, C_dense2_weight) + C_dense2_bias, name='CST_dense2')
    CST_dropout2 = get_dropout_layer(CST_dense2, 1, 'CST_dropout2')

    CTI_dense2 = tf.nn.relu(tf.matmul(CTI_dropout1, C_dense2_weight) + C_dense2_bias, name='CTI_dense2')
    CTI_dropout2 = get_dropout_layer(CTI_dense2, 1, 'CTI_dropout2')

    CTT_dense2 = tf.nn.relu(tf.matmul(CTT_dropout1, C_dense2_weight) + C_dense2_bias, name='CTT_dense2')
    CTT_dropout2 = get_dropout_layer(CTT_dense2, 1, 'CTT_dropout2')

with tf.variable_scope("common_layer_3"):
    C_dense3_num = 10
    C_dense3_weight = get_weight([C_dense2_num, C_dense3_num], 'C_dense3_weight')
    C_dense3_bias = get_bias([SI_hidden_shape[0], C_dense3_num], 'C_dense3_bias')

    CSI_dense3 = tf.nn.relu(tf.matmul(CSI_dropout2, C_dense3_weight) + C_dense3_bias, name='CSI_dense3')
    CSI_dropout3 = get_dropout_layer(CSI_dense3, 1, 'CSI_dropout3')

    CST_dense3 = tf.nn.relu(tf.matmul(CST_dropout2, C_dense3_weight) + C_dense3_bias, name='CST_dense3')
    CST_dropout3 = get_dropout_layer(CST_dense3, 1, 'CST_dropout3')

    CTI_dense3 = tf.nn.relu(tf.matmul(CTI_dropout2, C_dense3_weight) + C_dense3_bias, name='CTI_dense3')
    CTI_dropout3 = get_dropout_layer(CTI_dense3, 1, 'CTI_dropout3')

    CTT_dense3 = tf.nn.relu(tf.matmul(CTT_dropout2, C_dense3_weight) + C_dense3_bias, name='CTT_dense3')
    CTT_dropout3 = get_dropout_layer(CTT_dense3, 1, 'CTT_dropout3')

CSI_output = CSI_dropout3
CST_output = CST_dropout3
CTI_output = CTI_dropout3
CTT_output = CTT_dropout3

# custom loss function

def euclidean_loss(mat1, mat2):
    with tf.variable_scope("euclidean_loss"):
        diff = tf.subtract(mat1, mat2)
        diff_squared = tf.square(diff)
        row_sum = tf.reduce_sum(diff_squared, 1)
        error = tf.reduce_mean(row_sum, 0)

        return error

def mmd_loss(mat1, mat2):
    with tf.variable_scope("mmd_loss"):
        diff = tf.subtract(mat1, mat2)
        mmd = tf.reduce_mean(diff, 0)
        error = tf.norm(mmd, ord='euclidean')

        return error

def cross_entropy_loss(mat1, mat2):
    with tf.variable_scope("cross_entropy_loss"):
        error = tf.nn.softmax_cross_entropy_with_logits(labels=mat1, logits=mat2)
        error = tf.reduce_mean(error, 0)

        return error    

learning_rate = 0.001

source_l1_loss = euclidean_loss(SI_hidden, ST_hidden)
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=source_l1_loss, global_step=tf.train.get_global_step())

target_l1_loss = euclidean_loss(TI_hidden, TT_hidden)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=target_l1_loss, global_step=tf.train.get_global_step())

image_l2_loss = mmd_loss(SI_hidden, TI_hidden)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=image_l2_loss, global_step=tf.train.get_global_step())

text_l2_loss = mmd_loss(ST_hidden, TT_hidden)
optimizer4 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=text_l2_loss, global_step=tf.train.get_global_step())

source_image_l3_loss = cross_entropy_loss(source_label_input, CSI_output)
optimizer5 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=source_image_l3_loss, global_step=tf.train.get_global_step())

source_text_l3_loss = cross_entropy_loss(source_label_input, CST_output)
optimizer6 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=source_text_l3_loss, global_step=tf.train.get_global_step())

target_image_l3_loss = cross_entropy_loss(target_label_input, CTI_output)
optimizer7 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=target_image_l3_loss, global_step=tf.train.get_global_step())

target_text_l3_loss = cross_entropy_loss(target_label_input, CTT_output)
optimizer8 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=target_text_l3_loss, global_step=tf.train.get_global_step())

writer = tf.summary.FileWriter('/tmp/tensorboard', graph=tf.get_default_graph())

init_op = tf.global_variables_initializer()

base_path = '/home/mayank/Desktop/BTP/Datasets/NUS_WIDE_10k/'
nus_wide_10k_loader.setup_batch(base_path, 0.90, 0.20)
def generate_next_batch(domain, kind):
    if domain=='source' and kind=='train':
        text_image_label = nus_wide_10k_loader.get_batch_source_train(batch_size)
    elif domain=='source' and kind=='test':
        text_image_label = nus_wide_10k_loader.get_batch_source_test(batch_size)
    elif domain=='target' and kind=='train': 
        text_image_label = nus_wide_10k_loader.get_batch_target_train(batch_size)
    elif domain=='target' and kind=='test':
        text_image_label = nus_wide_10k_loader.get_batch_target_test(batch_size)

    batch_image = np.zeros([batch_size, 256, 256, 3])
    batch_text = np.zeros([batch_size, 300])
    batch_label = np.zeros([batch_size, 10])
    counter = 0

    for text, image, label in text_image_label:
        batch_text[counter] = text

        temp_image = ndimage.imread(base_path+'Dataset/' + str(image))
        temp_image = misc.imresize(temp_image, (256, 256, 3))
        
        # misc.imshow(temp_image)

        batch_label[counter] = label

        counter+= 1

    return batch_text, batch_image, batch_label

train_epoch = 1
test_epoch = 1

with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(train_epoch):
        batch_source_text, batch_source_image, batch_source_label = generate_next_batch('source', 'train')
        #batch_target_text, batch_target_image, batch_target_label = generate_next_batch('target', 'train')

        # _, c = sess.run([optimizer1, source_l1_loss], feed_dict={source_image_input: batch_source_image, source_text_input: batch_source_text})
        # print "Epoch:", epoch, " Batch:", batch, " Source l1 loss =", c

        # _, c = sess.run([optimizer2, target_l1_loss], feed_dict={target_image_input: batch_target_image, target_text_input: batch_target_text})
        # print "Epoch:", epoch, " Batch:", batch, " Target l1 loss =", c

        # _, c = sess.run([optimizer3, image_l2_loss], feed_dict={source_image_input: batch_source_image, target_image_input: batch_target_image})
        # print "Epoch:", epoch, " Batch:", batch, " Image l2 loss =", c

        # _, c = sess.run([optimizer4, text_l2_loss], feed_dict={source_text_input: batch_source_text, target_text_input: batch_target_text})
        # print "Epoch:", epoch, "Text l2 loss =", c

        _, c = sess.run([optimizer5, source_image_l3_loss], feed_dict={source_image_input: batch_source_image, source_label_input: batch_source_label})
        print "Epoch:", epoch, " Source image l3 loss =", c

        _, c = sess.run([optimizer6, source_text_l3_loss], feed_dict={source_text_input: batch_source_text, source_label_input: batch_source_label})
        print "Epoch:", epoch, " Source text l3 loss =", c

        # _, c = sess.run([optimizer7, target_image_l3_loss], feed_dict={target_image_input: batch_target_image, target_label_input: batch_target_label})
        # print "Epoch:", epoch, " Target image l3 loss =", c

        # _, c = sess.run([optimizer8, target_text_l3_loss], feed_dict={target_text_input: batch_target_text, target_label_input: batch_target_label})
        # print "Epoch:", epoch, " Target text l3 loss =", c

        print

    """
    batch_source_text, batch_source_image, batch_source_label = generate_next_batch('source', 'test')
    batch_target_text, batch_target_image, batch_target_label = generate_next_batch('target', 'test')

    print "Test source image l3 loss =", sess.run(source_image_l3_loss, feed_dict={source_image_input: batch_source_image, source_label_input: batch_source_label})
    
    print "Test source text l3 loss =", sess.run(source_text_l3_loss, feed_dict={source_text_input: batch_source_text, source_label_input: batch_source_label})
    
    print "Test target image l3 loss =", sess.run(target_image_l3_loss, feed_dict={target_image_input: batch_target_image, target_label_input: batch_target_label})
    
    print "Test target text l3 loss =", sess.run(target_text_l3_loss, feed_dict={target_text_input: batch_target_text, target_label_input: batch_target_label})
    """

    file = open('tsne_labels.csv', 'w') 
    
    embedding_var = tf.Variable(tf.truncated_normal([1, 2048]), name='embedding')
    for epoch in range(test_epoch):
        batch_source_text, batch_source_image, batch_source_label = generate_next_batch('source', 'test')
        temp_var = sess.run(SI_hidden, feed_dict={source_image_input: batch_source_image, source_text_input: batch_source_text})
        embedding_var = tf.concat([embedding_var, temp_var], 0)
        lab = [np.where(r==1)[0][0] for r in batch_source_label]
        file.write(lab)

    file.close()

writer = tf.summary.FileWriter('/tmp/tensorboard', graph=tf.get_default_graph())
