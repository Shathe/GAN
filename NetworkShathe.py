import tensorflow as tf



class NetworkShathe:
    def __init__(self):
        self.layer_index = 0


    # 2D convolution
    def conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, last=False):
        self.layer_index = self.layer_index + 1

        with tf.variable_scope('conv2d_bn_'+str(self.layer_index)):

            # Bottleneck, reduce dimentionality (in conv blocks of densenet, the input dimensionality increses a lot)
            x = tf.layers.conv2d(x, filters*4, (1, 1), strides=(1, 1), padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='bottleneck'+ str(self.layer_index)) # there is also dilation_rate!
            # Convolution
            x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate, activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv'+ str(self.layer_index)) # there is also dilation_rate!
            if not last:
                # normalizatoin and activation
                x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(self.layer_index))
                x = tf.nn.leaky_relu(x, name='Lrelu'+ str(self.layer_index))
                #x = tf.nn.dropout(x, keep_prob=0.93)
            return x



    # 2D traspose-convolution
    def deconv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None):
        self.layer_index = self.layer_index + 1

        with tf.variable_scope('deconv2d_bn_'+str(self.layer_index)):
            print(x.get_shape())

            # Traspose convolution
            x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(self.layer_index), bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None) # there is also dilation_rate!
            # normalizatoin and activation
            x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(self.layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='Lrelu'+ str(self.layer_index))
            print(x.get_shape())
            #x = tf.nn.dropout(x, keep_prob=0.93)

            return x


    # Dense block of convoutions
    def concatenation_convs(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,   times=6, compression=0.40):
        self.layer_index = self.layer_index + 1
        print(x.get_shape())

        with tf.variable_scope('dense_block'+str(self.layer_index)):

            next_input = x
            for time in xrange(times):

                output =  self.conv2d_bn(next_input, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training)
                next_input = tf.concat([output, next_input], axis=3)


            # output dimensionality reduction
            if compression:
                filters_pre = next_input.get_shape()[3].value
                compresion_filters = int(compression * filters_pre)
                next_input =  self.conv2d_bn(next_input, compresion_filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training)

            print(next_input.get_shape())

        return next_input


    # Last ense block of convoutions. It returns  all the layers to compute the loss with all of them
    def concatenation_convs_outputs(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,   times=6, n_classes=10):
        self.layer_index = self.layer_index + 1

        with tf.variable_scope('concat_'+str(self.layer_index)):
            outputs=[]
            next_input = x
            for time in xrange(times):

                output =  self.conv2d_bn(next_input, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training)
                next_input = tf.concat([output, next_input], axis=3)

                self.layer_index = self.layer_index + 1
                # output layer
                new_out =  self.conv2d_bn(output, n_classes, 1, 1, padding='same', strides=(1, 1), training=training, last=True)
                #add it to the next input
                next_input = tf.concat([new_out, next_input], axis=3)
                # concat outputs
                outputs = outputs + [new_out]



        return outputs

    # Bilinear resize
    def bilinear_resize(self, x, width, height, depth):
        x = tf.image.resize_bilinear(x, [width, height], align_corners=True)
        x.set_shape([None, width, height, depth])
        return x


    def net(self, input_x=None, n_classes=20, weights=None, width=384, height=384, channels=3, training=True):
        rgb = input_x # 256,256
        # Level 1
        enc_1_pool = self.conv2d_bn(input_x, 32, 3, 3, padding='same', strides=(2, 2), training=training) 

        # Concat the rgb input resized to the next  encoder level
        rgb_reduce_2 = self.bilinear_resize(rgb, int(height/2), int(width/2), channels) 
        concat_enc1_rgb = tf.concat([rgb_reduce_2, enc_1_pool], axis=3)

        # Level 2
        enc_2 =  self.concatenation_convs(concat_enc1_rgb, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=8)# 128,128
        enc_2_pool = tf.layers.average_pooling2d(enc_2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(self.layer_index)) # 64,64

        # Concat the rgb input resized to the next  encoder level
        rgb_reduce_4 = self.bilinear_resize(rgb, int(height/4), int(width/4), channels) 
        concat_enc2_rgb = tf.concat([rgb_reduce_4, enc_2_pool], axis=3)
        
        # Level 3
        enc_3 =  self.concatenation_convs(concat_enc2_rgb, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=16)# 64,64
        enc_3_pool = tf.layers.average_pooling2d(enc_3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(self.layer_index)) # 32,32



        # Concat the rgb input resized to the next  encoder level
        rgb_reduce_8 = self.bilinear_resize(rgb, int(height/8), int(width/8), channels) 
        concat_enc3_rgb = tf.concat([rgb_reduce_8, enc_3_pool], axis=3)
        
        # Level 4
        enc_4 =  self.concatenation_convs(concat_enc3_rgb, 12, 3, 3, padding='same', strides=(1, 1), training=training, times=24)# 32,32
        enc_4_pool = tf.layers.average_pooling2d(enc_4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(self.layer_index)) # 16, 16

        enc_4_dilated =  self.concatenation_convs(concat_enc3_rgb, 8, 3, 3, padding='same', dilation_rate=(3, 3), strides=(1, 1), training=training, times=24)# 32,32


        # Level 5
        enc_5 =  self.concatenation_convs(enc_4_pool, 12, 3, 3, padding='same', strides=(1, 1), training=training, times=16)# 16, 16
        enc_5_pool = tf.layers.average_pooling2d(enc_5, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(self.layer_index)) # 8,8

        enc_5_dilated =  self.concatenation_convs(enc_4_dilated, 8, 3, 3, padding='same', dilation_rate=(5, 5), strides=(1, 1), training=training, times=16)# 32,32

        # Level 6
        enc_6 =  self.concatenation_convs(enc_5_pool, 12, 3, 3, padding='same', strides=(1, 1), training=training, times=8)# 8,8
        enc_6_pool = tf.layers.average_pooling2d(enc_6, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(self.layer_index)) # 4,4

        enc_6_dilated =  self.concatenation_convs(enc_5_dilated, 8, 3, 3, padding='same', dilation_rate=(7, 7), strides=(1, 1), training=training, times=8)# 32,32

        # Level 7
        enc_7 =  self.concatenation_convs(enc_6_pool, 12, 3, 3, padding='same', strides=(1, 1), training=training, times=5)# 4, 4
        
        enc_7_dilated =  self.concatenation_convs(enc_6_dilated, 8, 3, 3, padding='same', dilation_rate=(9, 9), strides=(1, 1), training=training, times=5)# 32,32


        # Decoder

        # Level up 6 
        dec_6 =  self.deconv2d_bn(enc_7, 64, 3, 3, padding='same', strides=(2, 2), training=training)# 8,8
        x = tf.concat([enc_6, dec_6], axis=3)
        block_dec_6 =  self.concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=8)

        # Level up 5
        dec_5 =  self.deconv2d_bn(block_dec_6, 96, 3, 3, padding='same', strides=(2, 2), training=training)# 16, 16
        x = tf.concat([enc_5, dec_5], axis=3)
        block_dec_5 =  self.concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=16)

        # Level up 4
        dec_4 =  self.deconv2d_bn(block_dec_5, 126, 3, 3, padding='same', strides=(2, 2), training=training)# 32,32
        dilateds = tf.concat([enc_7_dilated, enc_6_dilated, enc_5_dilated, enc_4_dilated], axis=3)
        dilateds = self.conv2d_bn(dilateds, int(dilateds.get_shape()[3].value/4), 1, 1, padding='same', strides=(1, 1), training=training) 

        x = tf.concat([enc_4, dec_4, rgb_reduce_8, dilateds], axis=3)
        block_dec_4 =  self.concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=24)

        # Level up 3
        dec_3 =  self.deconv2d_bn(block_dec_4, 156, 3, 3, padding='same', strides=(2, 2), training=training)# 64,64
        x = tf.concat([enc_3, dec_3, rgb_reduce_4], axis=3)
        block_dec_3 =  self.concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=12)

        # Level up 2
        dec_2 =  self.deconv2d_bn(block_dec_3, 186, 3, 3, padding='same', strides=(2, 2), training=training)# 112,112
        dec_2_concat = tf.concat([enc_2, dec_2, rgb_reduce_2, enc_1_pool], axis=3)

        '''
        Pyramid pooling combination
        '''

        # Decoders from other decoders levels. Bottlenecks
        dec_1_from_dec_3 = self.deconv2d_bn(dec_3, 64, 1, 1, padding='same', strides=(2, 2), training=training)
        dec_1_from_dec_4 = self.deconv2d_bn(dec_4, 32, 1, 1, padding='same', strides=(4 ,4), training=training)
        dec_1_from_dec_5 = self.deconv2d_bn(dec_5, 16, 1, 1, padding='same', strides=(8 ,8), training=training)
        dec_1_from_dec_6 = self.deconv2d_bn(dec_6, 8, 1, 1, padding='same', strides=(16 ,16), training=training)
        decs_concat = tf.concat([ dec_1_from_dec_3, dec_1_from_dec_4, dec_1_from_dec_5, dec_1_from_dec_6], axis=3)


        x = dec_2_concat
        filters_per_pool = 96
        # Different poolings from  the last traspose convolution layer
        global_pooling = tf.reduce_mean(x, [1,2])
        global_pooling_dilated = tf.reduce_mean(dilateds, [1,2])
        global_pooling = tf.concat([global_pooling, global_pooling_dilated], axis=1)


        denso = tf.layers.dense(global_pooling,global_pooling.get_shape()[1].value)
        denso_pool = tf.reshape(denso,[tf.shape(global_pooling)[0], 1, 1, global_pooling.get_shape()[1].value])
        denso_pool = tf.image.resize_bilinear(denso_pool, [height/2, width/2], align_corners=True)

        pool_1 = tf.reshape(global_pooling,[tf.shape(global_pooling)[0], 1, 1, global_pooling.get_shape()[1].value])
        pool_2 = tf.layers.average_pooling2d(x, pool_size=(56, 56), strides=(56,56))
        pool_3 = tf.layers.average_pooling2d(x, pool_size=(28,28), strides=(28,28))
        pool_4 = tf.layers.average_pooling2d(x, pool_size=(14,14), strides=(14,14))

        # Different poolings from  the dilated convolution layers
        pool_5 = self.conv2d_bn(dilateds, filters_per_pool, 1, 1, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
        pool_6 = self.conv2d_bn(dilateds, filters_per_pool, 3, 3, padding='same', strides=(1, 1), dilation_rate=(3, 3), training=training)
        pool_7 = self.conv2d_bn(dilateds, filters_per_pool, 3, 3, padding='same', strides=(1, 1), dilation_rate=(5, 5), training=training)
        pool_8 = self.conv2d_bn(dilateds, filters_per_pool, 3, 3, padding='same', strides=(1, 1), dilation_rate=(7, 7), training=training)
        pools_dilated = tf.concat([pool_5,pool_6, pool_7, pool_8 ], axis=3)
        pools_dilated = self.deconv2d_bn(pools_dilated, filters_per_pool*4, 1, 1, padding='same', strides=(4, 4), training=training)

        # Resize the poolings
        pool_1 = self.conv2d_bn(pool_1, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training)
        pool_1 = tf.image.resize_bilinear(pool_1, [height/2, width/2], align_corners=True)
        pool_2 = self.conv2d_bn(pool_1, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training)
        pool_2 = tf.image.resize_bilinear(pool_1, [height/2, width/2], align_corners=True)
        pool_3 = self.conv2d_bn(pool_1, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training)
        pool_3 = tf.image.resize_bilinear(pool_1, [height/2, width/2], align_corners=True)
        pool_4 = self.conv2d_bn(pool_1, filters_per_pool, 1, 1, padding='same', strides=(1, 1), training=training)
        pool_4 = tf.image.resize_bilinear(pool_1, [height/2, width/2], align_corners=True)

        pools = tf.concat([pool_1,pool_2,pool_3,pool_4, pools_dilated, denso_pool ], axis=3)


        pools = self.conv2d_bn(pools, int(pools.get_shape()[3].value/6), 1, 1, padding='same', strides=(1, 1), training=training) 
        decs_concat = self.conv2d_bn(decs_concat, int(decs_concat.get_shape()[3].value/4), 1, 1, padding='same', strides=(1, 1), training=training) 


        # concat pyramid pooling information with the last traspose convolutoin layer with the RGb informatpon
        x = tf.concat([dec_2_concat, decs_concat, pools, rgb_reduce_2], axis=3)

        # Last conv block
        x =  self.concatenation_convs(x, 24, 3, 3, padding='same', strides=(1, 1), training=training, times=8)
        # Conv blocks with several losses
        outputs = self.concatenation_convs_outputs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, times=3, n_classes=n_classes)

        print(x.get_shape())

        # Resize from 112 to 224 the outputs to compute the loss
        resize_outputs = []
        for output in outputs:
            resize_outputs = resize_outputs + [tf.image.resize_images( output, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )]

        return resize_outputs
        


