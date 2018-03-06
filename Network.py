import tensorflow as tf



def complex(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = increment(0)
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_bn(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x4 = concatenation_convs2(x, 36, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = increment(layer_index)
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    x3 = concatenation_convs2(x, 32, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = increment(layer_index)
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    x2 = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=20)
    layer_index = increment(layer_index)
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    x1 = concatenation_convs2(x, 32, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=8)
    layer_index = increment(layer_index)
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)

    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs2(x, 36, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = increment(layer_index)

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs2(x, 30, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = increment(layer_index)
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs2(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=20)
    layer_index = increment(layer_index)
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs2(x, 32, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=8)
    layer_index = increment(layer_index)
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)




    x3_ = deconv_simple(x3_, 128, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x4_ = deconv_simple(x4_, 92, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x5_ = deconv_simple(x5_, 64, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x6_ = deconv_simple(x6_, 32, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())

    layer_index = increment(layer_index)
    x = conv2d_bn_last(x, n_classes, 3, 3, padding='same', strides=(1, 1), training=training,layer_index=layer_index)

    print(x.get_shape())

    return x
#falta skip conections





def conv2d_simple(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=0, batch_norm=False, regularizers=None, bias=None):
    with tf.variable_scope('conv2d_bn_'+str(layer_index)):


        #x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
        # kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!
        filers_a = filters - int(filters/8)
        filers_b = int(filters/8)
        a = tf.layers.conv2d(x, filers_a, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!
        b = tf.layers.conv2d(x, filers_b, (num_row, num_col), strides=strides, padding=padding, dilation_rate=(5,5),activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv2'+ str(layer_index)) # there is also dilation_rate!
        x = tf.concat([a,b], axis=3)
        
        if not last:
            x = tf.layers.batch_normalization(x, training=training, name='BN'+ str(layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='relu'+ str(layer_index))
        return x



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):

        filers_a = filters - int(filters/6) - int(filters/8)
        filers_c = int(filters/8)
        filers_b = int(filters/6)

        x1 = tf.layers.conv2d(x, filers_a, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!

        x2 = tf.layers.conv2d(x, filers_c, (5,5), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv2'+ str(layer_index)) # there is also dilation_rate!

        x3 = tf.layers.conv2d(x, filers_b, (num_row, num_col), padding=padding, dilation_rate=(5,5),activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv3'+ str(layer_index)) # there is also dilation_rate!
        x3 = tf.layers.conv2d(x3, filers_b, (num_row, num_col),strides=strides,  padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv4'+ str(layer_index)) # there is also dilation_rate!
        x = tf.concat([x1, x2, x3], axis=3)

        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
       #  x = tf.nn.dropout(x, 0.25)
# tf.contrib.layers.l2_regularizer( scale=0.08)
        return x


def conv2d_bn_last(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, layer_index=0):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):


        x1 = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!


        return x1



def deconv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.variable_scope('deconv2d_bn_'+str(layer_index)):
        print('deconv')
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index), bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x

def concatenation_convs2(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, times=6):
    with tf.variable_scope('concat_'+str(layer_index)):

        concatetation_of_inputs = [x]
        previous_node = x
        for time in xrange(times):

            next_x = conv2d_bn(previous_node, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time)
            concatetation_of_inputs.append(next_x)

            previous_node = next_x

            #if time % 2 == 1:
            for elem in concatetation_of_inputs:
                previous_node = tf.concat([elem, previous_node], axis=3)
        
        filters=int(filters*times*0.50)
        next_x = conv2d_bn(previous_node, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time+1)

        print(next_x.get_shape())

    return next_x


def increment(layer_index):
    return layer_index + 1


def deconv_simple(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.variable_scope('deconv2d_bn_'+str(layer_index)):

        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index), bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x ,training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='relu'+ str(layer_index))
        return x




def simple(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = increment(0)
    # a layer instance is callable on a tensor, and returns a tensor
    layer_index = increment(layer_index)
    x = conv2d_simple(input_x, 64, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x6 = conv2d_simple(x, 64, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    x = tf.layers.average_pooling2d(x6, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x5 = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    x = tf.layers.average_pooling2d(x5, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x4 = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x3 = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x2 = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    print(x.get_shape())
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    

    x = deconv_simple(x, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x2], axis=3)
    print(x.get_shape())
    x = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x2_ = conv2d_simple(x, 128, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)

    x = deconv_simple(x2_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x3], axis=3)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x3_ = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)


    x = deconv_simple(x3_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x4], axis=3)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x4_ = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)


    x = deconv_simple(x, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x5], axis=3)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x5_ = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)


    x = deconv_simple(x5_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.concat([x, x6], axis=3)
    print(x.get_shape())
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_simple(x, 96, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)



    x5_ = deconv_simple(x5_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x4_ = deconv_simple(x4_, 96, 3, 3, padding='same', strides=(4, 4), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x3_ = deconv_simple(x3_, 128, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x2_ = deconv_simple(x2_, 128, 3, 3, padding='same', strides=(16,16), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)

    x = tf.concat([x, x2_, x3_, x4_, x5_], axis=3)
    print(x.get_shape())

    '''
    x = conv2d_simple(x, int(x.get_shape()[3]/5), 3, 3, padding='same', strides=(1, 1), training=training, last=True, layer_index=layer_index)
    layer_index = increment(layer_index)
    print(x.get_shape())
    '''
    x = conv2d_simple(x, n_classes, 3, 3, padding='same', strides=(1, 1), training=training, last=True, layer_index=layer_index)
    print(x.get_shape())
    return x

# complex Accuracy total: 0.782648680291974 test
# train entre 0.87-0.94

#simple Accuracy total: 0.690694763680896
# train 00.8-0.91
