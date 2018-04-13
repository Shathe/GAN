import tensorflow as tf


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

def small(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):


    layer_index = 0

    x1 = tf.layers.separable_conv2d(input_x, 32, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
    pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 112


    x2 = tf.layers.separable_conv2d(x1, 64, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
    pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 56

    x3 = tf.layers.separable_conv2d(x2, 96, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu)#salida 28

    x4 = tf.layers.separable_conv2d(x3, 172, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 14 

    x5 = tf.layers.separable_conv2d(x4, 712, (3, 3), strides=(2, 2), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 7 

    x6 = tf.layers.separable_conv2d(x5, 712, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) # 7 


    #x6 = tf.layers.separable_conv2d(x5, 256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), activation=tf.nn.selu) # 7
    # conv2d_simple(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, layer_index=0,last =False)
    x1_ = tf.image.resize_bilinear(x6, [14, 14], align_corners=True)

    x2_ = tf.layers.separable_conv2d(x1_, 172, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) 
    x2_ = tf.add(x4, x2_)


    x3_ = tf.image.resize_bilinear(x2_, [56, 56], align_corners=True)
    print(x2.shape)

    print(x3_.shape)
    #x4_ = tf.concat([x3_, x6], axis=3)
    x4_ = tf.layers.separable_conv2d(x3_, 64, (3, 3), strides=(1, 1), padding='same', depthwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
     pointwise_initializer=tf.contrib.layers.xavier_initializer_conv2d(),  activation=tf.nn.selu) 
    x4_ = tf.add(x4_, x2)


    x5_ = conv2d_simple(x4_, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=12, last=True)
    output = tf.image.resize_bilinear(x5_, [224, 224], align_corners=True)


    return output



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


