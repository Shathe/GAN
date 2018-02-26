import tensorflow as tf


def simple(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = increment(0)
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_bn(input_x, 32, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x4 = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = increment(layer_index)
    print(x4.get_shape())
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x3 = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = increment(layer_index)
    print(x3.get_shape())
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x2 = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=18)
    layer_index = increment(layer_index)
    print(x2.get_shape())
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x1 = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=9)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())

    print('-------------')
    

    x = deconv2d_bn(x, 64, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x1], axis=3)

    print(x.get_shape())
    x = concatenation_convs2(x, 32, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=4)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = deconv2d_bn(x, 84, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x2], axis=3)

    print(x.get_shape())
    x = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=9)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = deconv2d_bn(x, 100, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x3], axis=3)
    print(x.get_shape())
    x = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = deconv2d_bn(x, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x4], axis=3)
    print(x.get_shape())
    x = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = deconv2d_bn(x, 148, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, n_classes*10, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, n_classes, 3, 3, padding='same', strides=(1, 1), training=training, last=True, layer_index=layer_index)

    return x

#falta skip conections


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=0):
    with tf.variable_scope('conv2d_bn_'+str(layer_index)):

        x = tf.layers.conv2d(x, filters*4, (num_row, num_col), strides=(1,1), padding=padding, dilation_rate=dilation_rate,activation=tf.nn.softmax,   bias_initializer=tf.contrib.layers.xavier_initializer(),
                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(), name='conv_bottle'+ str(layer_index)) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x, axis=3, training=training, name='BN_bottle'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu_bottle'+ str(layer_index))

        n_filter=int(filters/3)
        x1 = tf.layers.conv2d(x, n_filter, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=tf.nn.softmax,   bias_initializer=tf.contrib.layers.xavier_initializer(),
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(), name='conv1'+ str(layer_index)) # there is also dilation_rate!

        x2 = tf.layers.conv2d(x, n_filter, (5,5), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=tf.nn.softmax,   bias_initializer=tf.contrib.layers.xavier_initializer(),
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(), name='conv2'+ str(layer_index)) # there is also dilation_rate!

        x3 = tf.layers.conv2d(x, n_filter, (num_row, num_col), padding=padding, dilation_rate=(5,5),activation=tf.nn.softmax,   bias_initializer=tf.contrib.layers.xavier_initializer(),
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(), name='conv3'+ str(layer_index)) # there is also dilation_rate!
        x3 = tf.layers.conv2d(x3, n_filter, (num_row, num_col),strides=strides,  padding=padding, dilation_rate=dilation_rate,activation=tf.nn.softmax,   bias_initializer=tf.contrib.layers.xavier_initializer(),
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(), name='conv4'+ str(layer_index)) # there is also dilation_rate!
        x = tf.concat([x1, x2, x3], axis=3)

        if not last:
            x = tf.layers.batch_normalization(x, axis=3, training=training, name='BN'+ str(layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
        return x

def deconv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.variable_scope('deconv2d_bn_'+str(layer_index)):

        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index), bias_initializer=tf.contrib.layers.xavier_initializer(),
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=tf.contrib.layers.l1_l2_regularizer()) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x, axis=3, training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
        return x

def concatenation_convs2(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=0, times=6):
    with tf.variable_scope('concat_'+str(layer_index)):

        concatetation_of_inputs = [x]
        previous_node = x
        for time in xrange(times):

            next_x = conv2d_bn(previous_node, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, last=last, layer_index=layer_index+time)
            concatetation_of_inputs.append(next_x)

            previous_node = next_x

            #if time % 2 == 1:
            for elem in concatetation_of_inputs:
                previous_node = tf.concat([elem, previous_node], axis=3)
        filters=int(filters*times*0.5)
        next_x = conv2d_bn(previous_node, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, last=last, layer_index=layer_index+time+1)

    return next_x


def increment(layer_index):
    return layer_index + 1
