from __future__ import division
import tensorflow as tf
import sys
sys.path.append("./Semantic-Segmentation-Suite/models")
import tensorflow.contrib.slim as slim

import resnet_v2
from DeepLabV3_plus import build_deeplabv3_plus
from PSPNet import build_pspnet



# USEFUL LAYERS
winit = tf.contrib.layers.xavier_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer


'''

LA ELECCION SE HACE EN BAS EA QUE, EL ESTADO DEL ARTE DE SEG. SEMAN. FCDENSENET, PSPNET, DEEPLAB.. 
LAS QUE SE BASAN MAS EN EL ENCOEDER, Y POR LO TANTO AFECTA MAS EL PRETARINING SON PSPNET Y SOBRETOOD DEEPLAB Y AMBAS USAN RESNET DE BASE
'''


#https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/DeepLabV3_plus.py

# Optimal image size (331, 331, 3). Minimum 224,224
def encoder_resnet101(input_x=None, n_classes=20, is_training=True):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=1e-5)):
        logits, end_points = resnet_v2.resnet_v2_101(input_x, is_training=is_training, scope='resnet_v2_101')
        x = tf.keras.layers.GlobalAveragePooling2D()(logits)
        x = tf.keras.layers.Dense(n_classes, activation=None, name='predictions_no_softmax')(x)
        '''
        same on tensorflow pure
        # Global average pooling
        x = tf.reduce_mean(x6_, [1,2])
        # Last layer 
        x = tf.layers.dense(x, n_classes)
        '''
        return x


def encoder_resnet101_decoder_deeplabv3plus(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    network, init_fn = build_deeplabv3_plus(input_x, preset_model = "DeepLabV3_plus-Res101", num_classes=n_classes, is_training=training )
    return network

def encoder_resnet101_decoder_pspnet(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    network, init_fn = build_pspnet(input_x, label_size=[height, width], num_classes=n_classes, preset_model="PSPNet-Res50", pooling_type = "AVG", weight_decay=1e-5, upscaling_method="conv", is_training=training)
    return network





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






