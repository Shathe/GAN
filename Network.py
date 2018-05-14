from __future__ import division
import tensorflow as tf
import os,time,cv2
import tensorflow.contrib.slim as slim
import numpy as np

# USEFUL LAYERS
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv2d
deconv = tf.contrib.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer

def module2(inputs, filters, name, training=True):

    with tf.variable_scope('module_' + name):
        filters_4 = int(filters/4)

        x_inputs = tf.layers.separable_conv2d(inputs, int(filters/4), (1,1), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(1,1))


        x1 = tf.layers.separable_conv2d(x_inputs, int(filters_4), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(1,1))

        x1 = tf.add(x1, x_inputs)

        x2 = tf.layers.separable_conv2d(x1, int(filters_4), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(2, 2))
        x2 = tf.add(x1, x2)
        x3 = tf.layers.separable_conv2d(x2, int(filters_4), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(4, 4))
        x3 = tf.add(x2, x3)
        x4 = tf.layers.separable_conv2d(x3, int(filters_4), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(8, 8))


        x = tf.concat((x1, x2, x3, x4), axis=3)

        # x = tf.add(x, x2)
        x = tf.layers.dropout(x, rate=0.30, training=training)


        x = tf.add(x, inputs[:,:,:,:filters])


        return x

def module3(inputs, filters, name, training=True):


    with tf.variable_scope('module_' + name):




        x = tf.layers.separable_conv2d(inputs, int(filters), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(2, 2))

        x2 = tf.layers.separable_conv2d(tf.add(x, inputs), int(filters), (3,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(1, 1))


        x = tf.layers.dropout(x2, rate=0.30, training=training)


        x = tf.add(x, inputs[:,:,:,:filters])


        return x
def module(inputs, filters, name, dilation_rate=(1, 1), training=True):


    with tf.variable_scope('module_' + name):




        x = tf.layers.separable_conv2d(inputs, int(filters), (1,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(1,1))

        x2 = tf.layers.separable_conv2d(x, int(filters), (3,1), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=(1,1))

        x = tf.layers.separable_conv2d(x2, int(filters), (3,1), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=dilation_rate)

        x = tf.layers.separable_conv2d(x, int(filters), (1,3), padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu, dilation_rate=dilation_rate)

        x = tf.add(x, x2)/2
        x = tf.layers.dropout(x, rate=0.30, training=training)


        x = tf.add(x, inputs[:,:,:,:filters])/2


        return x


def downsampling(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True):
    with tf.variable_scope('downsampling_' + name):
        x = tf.layers.separable_conv2d(inputs, filters, kernels, strides=strides, padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu)
        x2 = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2))
        x = tf.concat((x, x2), axis=3)

        return x


def downsampling2(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True):
    with tf.variable_scope('downsampling_' + name):
        x = tf.layers.separable_conv2d(inputs, filters, kernels, strides=strides, padding='same', 
            depthwise_initializer=winit, pointwise_initializer=winit,  activation=tf.nn.selu)

        return x

def upsampling(inputs, name, size_multiplier=2, training=True, last=False): 
     with tf.variable_scope('upsampling_' + name):
        x = tf.image.resize_bilinear(inputs, [inputs.get_shape()[1].value*size_multiplier, inputs.get_shape()[2].value*size_multiplier], align_corners=True)


        return x

def upsampling2(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True, last=False): 
     with tf.variable_scope('upsampling_' + name):

        activation = tf.nn.selu
        if last:
            activation=None

        x = tf.layers.conv2d_transpose(inputs, filters, kernels, strides=strides, padding='same', kernel_initializer=winit,  activation=activation) # there is also dilation_rate!

        return x


def MiniNet2(input_x=None, n_classes=20, training=True):
    d1 = downsampling(input_x, 12, 'd1', strides=(2, 2), kernels=(3, 3), training=training)
    d2 = downsampling(d1, 24, 'd2', strides=(2, 2), kernels=(3, 3), training=training)
    m1 = module(d2, 24, 'm2',dilation_rate=(1, 1), training=training)
    d3 = downsampling(m1, 48, 'd2_1', strides=(2, 2), kernels=(3, 3), training=training)
    m2 = module(d3, 48, 'm2m2',  dilation_rate=(1, 1), training=training)
    m3 = module(m2, 48, 'm2mm22',  dilation_rate=(1, 1), training=training)

    m3_ = module(m3, 48, 'm3_',  dilation_rate=(2, 2), training=training)
    m4_ = module(m3_, 48, 'm4_',  dilation_rate=(6, 6), training=training)
    m5_ = module(m4_, 48, 'm5_',  dilation_rate=(12, 12), training=training)

    d4 = downsampling(m5_, 96, 'd3', strides=(2, 2), kernels=(3, 3), training=training)
    m4 = module(d4, 96, 'm3',  dilation_rate=(4, 4), training=training)
    m5 = module(m4, 96, 'm4',  dilation_rate=(8, 8), training=training)
    m6 = module(m5, 96, 'm5',  dilation_rate=(2, 2), training=training)
    m7 = module(tf.add(m6, m4), 96, 'm6',  dilation_rate=(6, 6), training=training)
    up1 = upsampling2(tf.add(m7, m5)/2,48, 'up1', training=training)

    m8 = module(tf.add(up1, m3_)/2, 48, 'm7',  dilation_rate=(2, 2), training=training)
    m9 = module(tf.add(m8, m4_)/2, 48, 'm8',  dilation_rate=(4, 4), training=training)
    m10 = module(tf.add(m9, m5_)/2, 48, 'm8s',  dilation_rate=(8, 8), training=training)


    up2 = upsampling2(m10,24, 'up2', training=training)
    m12 = module(tf.add(up2, m1), 24, 'm10up2', training=training)
    up3 = upsampling2(m12,15, 'up22', training=training)
    m13 = module(tf.add(up3, d1), 15, 'm10', training=training)
    return   upsampling2(m13,n_classes, 'up3', training=training, last=True)


def MiniNet(input_x=None, n_classes=20, training=True):
    d1 = downsampling2(input_x, 12, 'd1', strides=(2, 2), kernels=(3, 3), training=training)
    d2 = downsampling2(d1, 24, 'd2', strides=(2, 2), kernels=(3, 3), training=training)
    d3 = downsampling2(d2, 48, 'd2_1', strides=(2, 2), kernels=(3, 3), training=training)
    d4 = downsampling2(d3, 96, 'd3', strides=(2, 2), kernels=(3, 3), training=training)
    m4 = module(d4, 96, 'm3',  dilation_rate=(4, 4), training=training)
    m5 = module(m4, 96, 'm4',  dilation_rate=(8, 8), training=training)
    m6 = module(m5, 96, 'm5',  dilation_rate=(2, 2), training=training)
    m7 = module(tf.add(m6, m4), 96, 'm6',  dilation_rate=(6, 6), training=training)
    up1 = upsampling2(tf.add(m7, m5),48, 'up1', training=training)

    d5 = downsampling2(d4, 192, 'd5', strides=(2, 2), kernels=(3, 3), training=training)
    d5 = module(d5, 192, 'm7d5',  dilation_rate=(2, 2), training=training)
    d6 = downsampling2(d5, 386, 'd6', strides=(2, 2), kernels=(3, 3), training=training)
    d6 = module(d6, 386, 'm7d6',  dilation_rate=(1, 1), training=training)
    d6 = module(d6, 386, 'm7d6d6',  dilation_rate=(1, 1), training=training)
    up_1 = upsampling2(d6,192, '_up11', training=training)
    up_1 = module(up_1, 192, 'm7d5up_1',  dilation_rate=(2, 2), training=training)
    up_2 = upsampling2(up_1,96, '_up22', training=training)
    up_3 = upsampling2(up_2,48, '_up33', training=training)


    up_concat = tf.concat((up_3, up1), axis=3)

    m9 = module(up_concat, 48, 'm8',  dilation_rate=(2, 2), training=training)


    up2 = upsampling2(tf.concat((m9, d3), axis=3),32, 'up2', training=training)
    up3 = upsampling2(tf.concat((up2, d2), axis=3),16, 'up22', training=training)

    out = upsampling2(tf.concat((up3, d1), axis=3),n_classes, 'up3', training=training, last=True)
    return   out



    '''

    up1 = upsampling(m6_add, 'up1', size_multiplier=2, training=training)
    m9 = module(up1, 42, 'm9', kernels=(3, 3), training=training)
    m9_add = tf.add(m9, m2)
    up2 = upsampling(m9_add, 'up2', size_multiplier=2, training=training)
    up2_concat = tf.concat([up2, d1], axis=3)

    last_layer = tf.layers.separable_conv2d(up2_concat, n_classes, (1, 1), padding='same',depthwise_initializer=winit, pointwise_initializer=winit)
    return upsampling(last_layer, 'last_layer', size_multiplier=2)
    '''



def small(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True, start_filters = 24):


    layer_index = 0

    x1 = tf.layers.separable_conv2d(input_x, start_filters, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
    pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 112
    x2 = tf.layers.separable_conv2d(x1, start_filters*4, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
    pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 56
    x3 = tf.layers.separable_conv2d(x2, start_filters*4, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 28
    x4 = tf.layers.separable_conv2d(x3, start_filters*8, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 28
    x5= tf.layers.separable_conv2d(x4, start_filters*8, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 14 
    x6 = tf.layers.separable_conv2d(x5, start_filters*16, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 14 
    x7 = tf.layers.separable_conv2d(x6, start_filters*16, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 7 
    x8 = tf.layers.separable_conv2d(x7, start_filters*32, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 7 






    #join residual connections
    x8_res = tf.image.resize_bilinear(x8[:,:,:,:start_filters*8], [14, 14], align_corners=True)
    x7_res = tf.image.resize_bilinear(x7[:,:,:,:start_filters*8], [14, 14], align_corners=True) 
    x6_res = tf.image.resize_bilinear(x6[:,:,:,:start_filters*8], [14, 14], align_corners=True)
    x5_res = tf.image.resize_bilinear(x5, [14, 14], align_corners=True)
    x4_res = tf.image.resize_bilinear(x4, [14, 14], align_corners=True)

    res_1 = tf.add(x8_res, x6_res)
    res_2 = tf.add(x7_res, x5_res)
    res_3 = tf.add(res_1, x4_res)
    res = tf.add(res_3, res_2)

    # up
    res_14 = tf.layers.separable_conv2d(res, start_filters*8, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) 

    x2_up = tf.image.resize_bilinear(x8, [14, 14], align_corners=True)
    x3_up = tf.layers.separable_conv2d(x2_up, start_filters*8, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) 
    x4_up = tf.add(x3_up, res_14)
    x5_up = tf.image.resize_bilinear(x4_up, [56, 56], align_corners=True)
    x6_up = tf.layers.separable_conv2d(x5_up, start_filters*8, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) 
    res_56 = tf.image.resize_bilinear(res_14, [56, 56], align_corners=True)
    x7_up = tf.add(res_56, x6_up)
    last_layer = conv2d_simple(x7_up, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=12, last=True)
    output = tf.image.resize_bilinear(last_layer, [224, 224], align_corners=True)


    return output



def encoder_classif(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):


    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    # Global average pooling
    x = tf.reduce_mean(x6_, [1,2])
    # Last layer 
    x = tf.layers.dense(x, n_classes)

    return x





def nasnet(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    net = tf.keras.applications.NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_x, pooling=None, classes=n_classes)
    print(net.summary())
    output = net.output

    # get same 14x14 and 56x56 of the encoder to make skip conections
    ac1 = net.get_layer('activation_15').output
    ac2 = net.get_layer('activation_13').output
    ac3 = net.get_layer('activation_19').output
    ac4 = net.get_layer('adjust_relu_1_0').output
    ac_56_56 = tf.concat([ac4, ac2, ac1, ac3], axis=3)

    ac5 = net.get_layer('activation_130').output
    ac6 = net.get_layer('activation_129').output
    ac7 = net.get_layer('activation_94').output
    ac8 = net.get_layer('adjust_relu_1_9').output
    ac_14_14 = tf.concat([ac5, ac6, ac7, ac8], axis=3)
    output_14_14 = tf.image.resize_bilinear(output, [14, 14], align_corners=True)
    size_14_14 = tf.concat([ac_14_14, output_14_14], axis=3)
    size_14_14 = bottleneck(size_14_14, 256, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=2)

    # Here there are size_14_14 and ac_56_56 layers.


    # Some convolutions
    size_14_14 = conv2d_sep_bn(size_14_14, 256, 3, 3, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=4 )
    size_14_14 = conv2d_sep_bn(size_14_14, 256, 3, 3, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=5 )
    size_14_14 = conv2d_sep_bn(size_14_14, 256, 3, 3, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=45 )

    size_56_56 = tf.image.resize_bilinear(size_14_14, [56, 56], align_corners=True)
    size_56_56 = tf.concat([size_56_56, ac_56_56], axis=3)
    size_56_56 = bottleneck(size_56_56, 128, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=55)

    # Now only 56x56 and again some convolutoins
    size_56_56 = conv2d_sep_bn(size_56_56, 128, 3, 3, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=6 )
    size_56_56 = conv2d_sep_bn(size_56_56, 128, 3, 3, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=7 )

    # This is a Pyramide pooling module

    x = size_56_56
    global_pooling = tf.reduce_mean(x, [1,2])

    pool_1 = tf.reshape(global_pooling,[tf.shape(global_pooling)[0], 1, 1, global_pooling.get_shape()[1].value])
    pool_1 = bottleneck(pool_1, 32, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=8)
    pool_2 = tf.layers.average_pooling2d(x, pool_size=(28, 28), strides=(28,28))
    pool_2 = bottleneck(pool_2, 32, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=9)
    pool_3 = tf.layers.average_pooling2d(x, pool_size=(7,7), strides=(7,7))
    pool_3 = bottleneck(pool_3, 32, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=10)
    pool_4 = tf.layers.average_pooling2d(x, pool_size=(14,14), strides=(14,14))
    pool_4 = bottleneck(pool_4, 32, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=11)

    pool_1 = tf.image.resize_bilinear(pool_1, [height, width], align_corners=True)
    pool_2 = tf.image.resize_bilinear(pool_2, [height, width], align_corners=True)
    pool_3 = tf.image.resize_bilinear(pool_3, [height, width], align_corners=True)
    pool_4 = tf.image.resize_bilinear(pool_4, [height, width], align_corners=True)


    # Resize the 56x56 into 224x224 and concatenate to the Pyramide pooling
    size_224_224 = tf.image.resize_bilinear(size_56_56, [height, width], align_corners=True)

    output = tf.concat([size_224_224, pool_1, pool_2, pool_3, pool_4], axis=3)

    #Last convolution of n_classes channels
    output = conv2d_simple(output, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=12, last=True)

    return output

def encoder_decoder_v0(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    layer_index = layer_index + 1
    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x





def encoder_decoder_v1(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())

    layer_index = layer_index + 1
    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x




def encoder_decoder_v2(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1



    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(2,2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(4,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(16,16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())
    x = conv2d_bn(x, 128, 1,1, padding='same', strides=(1, 1), training=training, layer_index=layer_index)

    layer_index = layer_index + 1
    x = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=10)
    layer_index = layer_index + 1

    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)
    x = tf.image.resize_images( x, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    print(x.get_shape())

    return x






def encoder_decoder_v3(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x6_ = conv2d_simple(x6_, 48, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    print('aaaaa')
    print(x6_.get_shape())
    shape =[tf.shape(x6_)[0],  x6_.get_shape()[1].value * x6_.get_shape()[2].value * x6_.get_shape()[3].value]
    print(shape)
    reshape_tensor = tf.reshape(x6_,shape)
    print('aaaaa')
    print(reshape_tensor.get_shape())
    denso = tf.layers.dense(reshape_tensor,256)
    denso = tf.layers.dense(denso,512)
    denso = tf.layers.dense(denso,1568)
    print(denso.get_shape())
    reshape_tensor_denso = tf.reshape(denso,[tf.shape(denso)[0], 7, 7, 32 ])



    x = deconv2d_bn(reshape_tensor_denso, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())

    layer_index = layer_index + 1
    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x




def encoder_decoder_v4(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())


    layer_index = layer_index + 1
    outputs= concatenation_convs_outputs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=10, n_classes=n_classes)



    print(x.get_shape())

    return outputs





def encoder_decoder_v5(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())
    layer_index = layer_index + 1

    filters_final = x.get_shape()[3].value
    filters_per_pool = int(filters_final/4)
    # aqui hacer los poolings difenretes
    pool_1_denso = tf.reduce_mean(x, [1,2])
    pool_1 = tf.reshape(pool_1_denso,[tf.shape(pool_1_denso)[0], 1, 1, pool_1_denso.get_shape()[1].value])

    pool_2 = tf.layers.average_pooling2d(x, pool_size=(112, 112), strides=(112,112), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    pool_3 = tf.layers.average_pooling2d(x, pool_size=(56,56), strides=(56,56), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    pool_4 = tf.layers.average_pooling2d(x, pool_size=(28,28), strides=(28,28), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    pool_1 = conv2d_simple(pool_1, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index)
    layer_index = layer_index + 1
    pool_2 = conv2d_simple(pool_2, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index)
    layer_index = layer_index + 1
    pool_3 = conv2d_simple(pool_3, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index)
    layer_index = layer_index + 1
    pool_4 = conv2d_simple(pool_4, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index)
    layer_index = layer_index + 1
    
    print('pools')
    print(pool_1.get_shape())
    print(pool_2.get_shape())
    print(pool_3.get_shape())
    print(pool_4.get_shape())   
    
    deconv_1 = deconv2d_bn(pool_1, filters_per_pool, 1, 1, padding='same', strides=(224, 224), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    deconv_2 = deconv2d_bn(pool_2, filters_per_pool, 2, 2, padding='same', strides=(112, 112), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    deconv_3 = deconv2d_bn(pool_3, filters_per_pool, 2, 2, padding='same', strides=(56, 56), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    deconv_4 = deconv2d_bn(pool_4, filters_per_pool, 2, 2, padding='same', strides=(28, 28), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
   
    print('pools')
    print(deconv_1.get_shape())
    print(deconv_2.get_shape())
    print(deconv_3.get_shape())
    print(deconv_4.get_shape())

    x = tf.concat([x, deconv_1,deconv_2,deconv_3,deconv_4], axis=3)

    layer_index = layer_index + 1
    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x



def encoder_decoder_v_final(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos
    rgb = input_x
    rgb_reduce_2 = bilinear_resize(rgb, int(height/2), int(width/2), channels) 
    rgb_reduce_4 = bilinear_resize(rgb, int(height/4), int(width/4), channels) 
    rgb_reduce_8 = bilinear_resize(rgb, int(height/8), int(width/8), channels) 
    


    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1

    # Concat the rgb input resized to the next  encoder level

    x = tf.concat([rgb_reduce_2, x], axis=3)

    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1

    x = tf.concat([rgb_reduce_4, x], axis=3)

    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1

    x = tf.concat([rgb_reduce_8, x], axis=3)

    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=32)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=6)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())

    layer_index = layer_index + 1
    x = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=10)
    layer_index = layer_index + 1

    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x



def conv2d_simple(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, layer_index=0,last =False):

    with tf.variable_scope('conv2d_simple'+str(layer_index)):


        x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!
        if not last:
            x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x







def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, dilated=True ):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):


        # Bottleneck
        x = tf.layers.conv2d(x, filters*4, (1, 1), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='convbottle'+ str(layer_index)) # there is also dilation_rate!



        if dilated:
            filers_a = filters - int(filters/6) - int(filters/8)
            filers_c = int(filters/8)
            filers_b = int(filters/6)

            x1 = tf.layers.conv2d(x, filers_a, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!

            x2 = tf.layers.conv2d(x, filers_c, (5,5), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv2'+ str(layer_index)) # there is also dilation_rate!

            x3 = tf.layers.conv2d(x, filers_b, (num_row, num_col), padding=padding, dilation_rate=(5,5),activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv3'+ str(layer_index)) # there is also dilation_rate!
            x = tf.concat([x1, x2, x3], axis=3)
        else:
            x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!


        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
       #  x = tf.nn.dropout(x, 0.25)
# tf.contrib.layers.l2_regularizer( scale=0.08)
        return x






def deconv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.variable_scope('deconv2d_bn_'+str(layer_index)):
        print('deconv')
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index), bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x

def concatenation_convs(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, times=6, compression=0.40):
    with tf.variable_scope('concat_'+str(layer_index)):

        next_input = x
        for time in xrange(times):

            output = conv2d_bn(next_input, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time)

            next_input = tf.concat([output, next_input], axis=3)


        if compression:
            filters_pre = next_input.get_shape()[3].value
            compresion_filters = int(compression * filters_pre)
            next_input = conv2d_bn(next_input, compresion_filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time+1)

        print(next_input.get_shape())

    return next_input



def concatenation_convs_outputs(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, times=6, n_classes=10):
    with tf.variable_scope('concat_'+str(layer_index)):
    	outputs=[]
        next_input = x
        for time in xrange(times):

            output = conv2d_bn(next_input, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time)

            next_input = tf.concat([output, next_input], axis=3)
            layer_index = layer_index + 1

            new_out = conv2d_simple(output, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

            outputs = outputs + [new_out]



    return outputs


def bilinear_resize( x, width, height, depth):
    x = tf.image.resize_bilinear(x, [width, height], align_corners=True)
    x.set_shape([None, width, height, depth])
    return x


# 2D convolution
def bottleneck(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False, layer_index=0):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):

        # Bottleneck, reduce dimentionality (in conv blocks of densenet, the input dimensionality increses a lot)
        x = tf.layers.conv2d(x, filters, (1, 1), strides=(1, 1), padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='bottleneck'+ str(layer_index)) # there is also dilation_rate!
        # Convolution
   
        # normalizatoin and activation
        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index))
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
        #x = tf.nn.dropout(x, keep_prob=0.93)
    return x




def conv2d_sep_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0 ):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):


        x = tf.layers.separable_conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate, activation=None,
         depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv1'+ str(layer_index)) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x




def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # Stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features


def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
  """
  with tf.name_scope(scope) as sc:
    # Upsample
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l

def build_fc_densenet(inputs, num_classes, preset_model='FC-DenseNet56', n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None):
    """
    Builds the FC-DenseNet model
    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
      stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net



def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Builds the conv block for MobileNets
    Apply successivly a 2D convolution, BatchNormalization relu
    """
    # Skip pointwise by setting num_outputs=Non
    net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Builds the Depthwise Separable conv block for MobileNets
    Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
    """
    # Skip pointwise by setting num_outputs=None
    net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None)

    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net))
    return net

def build_mobile_unet(inputs, preset_model, num_classes):

    has_skip = False
    if preset_model == "MobileUNet":
        has_skip = False
    elif preset_model == "MobileUNet-Skip":
        has_skip = True
    else:
        raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

    #####################
    # Downsampling path #
    #####################
    net = ConvBlock(inputs, 64)
    net = DepthwiseSeparableConvBlock(net, 64)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip_1 = net

    net = DepthwiseSeparableConvBlock(net, 128)
    net = DepthwiseSeparableConvBlock(net, 128)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip_2 = net

    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip_3 = net

    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip_4 = net

    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


    #####################
    # Upsampling path #
    #####################
    net = conv_transpose_block(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    if has_skip:
        net = tf.add(net, skip_4)

    net = conv_transpose_block(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 256)
    if has_skip:
        net = tf.add(net, skip_3)

    net = conv_transpose_block(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 128)
    if has_skip:
        net = tf.add(net, skip_2)

    net = conv_transpose_block(net, 128)
    net = DepthwiseSeparableConvBlock(net, 128)
    net = DepthwiseSeparableConvBlock(net, 64)
    if has_skip:
        net = tf.add(net, skip_1)

    net = conv_transpose_block(net, 64)
    net = DepthwiseSeparableConvBlock(net, 64)
    net = DepthwiseSeparableConvBlock(net, 64)

    #####################
    #      Softmax      #
    #####################
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net
