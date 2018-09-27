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






