import tensorflow as tf


def simple(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

    layer_index = increment(0)
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_bn(input_x, 32, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, 64, 3, 3, padding='same', strides=(1, 1), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    x = conv2d_bn(x, 128, 3, 3, padding='same', strides=(1, 1), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = increment(layer_index)
    # x = layers.Flatten()(x)
    x = deconv2d_bn(x, 128, 3, 3, padding='same', strides=(2, 2), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, 128, 3, 3, padding='same', strides=(1, 1), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = deconv2d_bn(x, 128, 3, 3, padding='same', strides=(2, 2), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, 128, 3, 3, padding='same', strides=(1, 1), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = deconv2d_bn(x, 128, 3, 3, padding='same', strides=(2, 2), layer_index=layer_index)
    layer_index = increment(layer_index)
    x = conv2d_bn(x, n_classes, 3, 3, padding='same', strides=(1, 1), last=True, layer_index=layer_index)

    return x

#falta skip conections


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, last=False, layer_index=0):
    with tf.name_scope('conv2d_bn_'):
        if last:
            x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, activation=tf.nn.softmax, name='deconv'+ str(layer_index)) # there is also dilation_rate!
        else:
            x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index)) # there is also dilation_rate!
            x = tf.layers.batch_normalization(x, axis=3, training=training, name='BN'+ str(layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
        return x

def deconv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.name_scope('deconv2d_bn_'):

        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index)) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x, axis=3, training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
        return x

def increment(layer_index):
    return layer_index + 1
'''
def node(x, nb_filter):

    tower_1 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_1 = conv2d_bn(tower_1, nb_filter, 3, 3, padding='same', strides=(1, 1))

    tower_2 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_2 = conv2d_bn(tower_2, nb_filter, 5, 5, padding='same', strides=(1, 1))

    output = layers.concatenate([tower_1, tower_2], axis=3)
    return output
    #ahora toca un denseblock con los nodos


def dense_block(x, nb_layers, nb_filter):
    #Hacer algo como contaenation imapres o pares para ver si se reduce el numero mucho?
    filter_augmenation_step = 4
    concatetation_of_inputs = x
    for i in range(nb_layers):
        next_node = node(concatetation_of_inputs, nb_filter)
        concatetation_of_inputs = layers.concatenate([concatetation_of_inputs, next_node], axis=3)
        previous_node = next_node
        nb_filter = nb_filter + filter_augmenation_step


    return concatetation_of_inputs #hacia la transition layer

def transition_block(x, nb_filter):

    x = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
'''