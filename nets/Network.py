from __future__ import division
import tensorflow as tf
import sys
# USEFUL LAYERS
winit = tf.contrib.layers.xavier_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer



def conv2d(x, filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), activation='relu', training=True, last =False):

    with tf.name_scope('conv'):
        x = tf.layers.conv2d(x, filters, filter_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
        	activation=None, kernel_initializer=winit, kernel_regularizer=l2_regularizer(0.0002))

        if not last:
	        x = tf.layers.batch_normalization(x,  training=training) 
	        '''
	        Activation fucntion
	        '''
	        if 'prelu' in  activation:
	        	x = tf.keras.layers.PReLU(x)
	        elif 'leakyrelu' in activation:
	        	x = tf.nn.leaky_relu(x)
	        elif 'relu' in activation:
	        	x = tf.nn.relu(x)

        return x

def deconv2d_bn(x, filters, filter_size, padding='same', strides=(1, 1), training=True, activation='relu'):

    with tf.name_scope('deconv'):

        x = tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding=padding, 
        	kernel_initializer=winit, activation=None, kernel_regularizer=l2_regularizer(0.0002)) 
        x = tf.layers.batch_normalization(x,  training=training) 
        '''
        Activation fucntion
        '''
        if 'prelu' in  activation:
        	x = tf.keras.layers.PReLU(x)
        elif 'leakyrelu' in activation:
        	x = tf.nn.leaky_relu(x)
        elif 'relu' in activation:
        	x = tf.nn.relu(x)


        return x



def encoder_decoder_example(input_x=None, n_classes=20, training=True):

    x1 = conv2d(input_x, 32, (3, 3), padding='same', strides=(2, 2), dilation_rate=(1, 1), training=training)
    x2 = conv2d(x1, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x3 = conv2d(x2, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x4 = conv2d(x3, 128, (3, 3), padding='same', strides=(2, 2), dilation_rate=(1, 1), training=training)
    x5 = conv2d(x4, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(2, 2), training=training)
    x6 = conv2d(x5, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(2, 2), training=training)
    x6 = tf.add(x6, x4)
    x7 = conv2d(x6, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x8 = conv2d(x7, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x9 = conv2d(x8, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x9 = tf.add(x6, x9)
    x10 = conv2d(x9, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x11 = conv2d(x10, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x12 = conv2d(x11, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x12 = tf.add(x12, x9)
    x12 = deconv2d_bn(x12, 128, (3, 3), padding='same', strides=(2, 2),  training=training)
    x13 = conv2d(x12, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x14 = conv2d(x13, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x14 = tf.add(x14, x3)
    x15 = deconv2d_bn(x14, 32, (3, 3), padding='same', strides=(2, 2),  training=training)
    x16 = conv2d(x15, n_classes, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training, last=True)

    return x16


from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing, regularizers
from tensorflow.keras  import backend as K
import tensorflow as tf
import numpy as np


def encoder_decoder_example3(input_x=None, n_classes=20, training=True):
    # IF YOU USE THIS, CHANGE THE PREPROCESSING
    '''
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=input_x,  pooling='avg')
    # bn_conv1/keras_learning_phase:0
    #x = tf.get_default_graph().get_tensor_by_name("bn_conv1/keras_learning_phase:0")
    #print(x)
    #x = tf.assign(x, concat, validate_shape=False) # We force TF, to skip the shape validation step

    # SET KERAS VARIABLE IS TRAINING
    x = model.outputs[0]
    e0 = tf.get_default_graph().get_tensor_by_name("activation/Relu:0")
    e1 = tf.get_default_graph().get_tensor_by_name("add_2/add:0")
    e2 = tf.get_default_graph().get_tensor_by_name("add_6/add:0")
    e3 = tf.get_default_graph().get_tensor_by_name("add_12/add:0")
    e4 = tf.get_default_graph().get_tensor_by_name("add_15/add:0")
    '''
    x = MnasNet(n_classes=n_classes, inputs=input_x, alpha=1)
    x12 = deconv2d_bn(x, 128, (3, 3), padding='same', strides=(2, 2),  training=training)
    x12 = conv2d(x12, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x12 = conv2d(x12, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x12 = conv2d(x12, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x12 = deconv2d_bn(x12, 128, (3, 3), padding='same', strides=(2, 2),  training=training)
    x12 = conv2d(x12, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x14 = conv2d(x12, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x15 = deconv2d_bn(x14, 32, (3, 3), padding='same', strides=(2, 2),  training=training)
    x16 = conv2d(x15, n_classes, (1, 1), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training, last=True)
    return x16




def MnasNet(n_classes=1000, inputs=None, alpha=1):

    x = conv_bn(inputs, 32 * alpha, 3, strides=2)
    x = sepConv_bn_noskip(x, 16 * alpha, 3, strides=1)
    # MBConv3 3x3
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=2, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
    # MBConv3 5x5
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=2, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
    # MBConv6 5x5
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 3x3
    x = MBConv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 5x5
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 3x3
    x = MBConv_idskip(x, filters=320, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)

    # FC + POOL
    x = conv_bn(x, filters=1152 * alpha, kernel_size=1, strides=1)

    return x


# Convolution with batch normalization
def conv_bn(x, filters, kernel_size, strides=1, alpha=1, activation=True):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        alpha: An integer which multiplies the filters dimensionality
        activation: A boolean which indicates whether to have an activation after the normalization
    # Returns
        Output tensor.
    """
    filters = _make_divisible(filters * alpha)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    if activation:
        x = layers.ReLU(max_value=6)(x)
    return x


# Depth-wise Separable Convolution with batch normalization
def depthwiseConv_bn(x, depth_multiplier, kernel_size, strides=1):
    """ Depthwise convolution
    The DepthwiseConv2D is just the first step of the Depthwise Separable convolution (without the pointwise step).
    Depthwise Separable convolutions consists in performing just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).

    This function defines a 2D Depthwise separable convolution operation with BN and relu6.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                               padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = layers.ReLU(max_value=6)(x)
    return x


def sepConv_bn_noskip(x, filters, kernel_size, strides=1):
    """ Separable convolution block (Block F of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
    x = conv_bn(x, filters=filters, kernel_size=1, strides=1)

    return x


# Inverted bottleneck block with identity skip connection
def MBConv_idskip(x_input, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1):
    """ Mobile inverted bottleneck convolution (Block b, c, d, e of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        alpha: An integer which multiplies the filters dimensionality
    # Returns
        Output tensor.
    """

    depthwise_conv_filters = _make_divisible(x_input.shape[3].value)
    pointwise_conv_filters = _make_divisible(filters * alpha)

    x = conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
    x = conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

    # Residual connection if possible
    if strides == 1 and x.shape[3] == x_input.shape[3]:
        return layers.add([x_input, x])
    else:
        return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v