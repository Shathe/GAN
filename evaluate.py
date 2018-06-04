import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import random
import sys
import math
import os
from utils.utils import preprocess
from erfnet import erfnetA, erfnetB, erfnetC, erfnetD
import argparse
import time
import tensorflow.contrib.slim as slim
import Network
import cv2
sys.path.append("../Semantic-Segmentation-Suite/models")
import resnet_v2 
from DeepLabV3_plus import build_deeplabv3_plus
from PSPNet import build_pspnet
random.seed(os.urandom(9))
#tensorboard --logdir=train:./logs/train,test:./logs/test/

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset to train", default='./text_test')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-4)
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=1)
parser.add_argument("--max_batch_size", help="batch_size", default=1)
parser.add_argument("--n_classes", help="number of classes to classify", default=2)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=400)
parser.add_argument("--width", help="width", default=512)
parser.add_argument("--height", help="height", default=256)
parser.add_argument("--save_model", help="save_model", default=1)
args = parser.parse_args()




# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model ))
init_batch_size = int(args.init_batch_size)
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
channels = int(args.dimensions)
change_lr_epoch = math.pow(min_learning_rate/init_learning_rate, 1.0/total_epochs)
change_batch_size = (max_batch_size - init_batch_size) / float(total_epochs - 1)



with tf.device('/cpu:0'):



	# For Batch_norm or dropout operations: training or testing
	training_flag = tf.placeholder(tf.bool)

	# Placeholder para las imagenes.
	x = tf.placeholder(tf.float32, shape=[1, height, width, channels], name='input')
	label = tf.placeholder(tf.float32, shape=[1, height, width, n_classes], name='output')
	mask_label = tf.placeholder(tf.float32, shape=[1, height, width, n_classes], name='mask')
	# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

	# Network
	# output = Network.encoder_decoder_v1(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)

	# Network
	'''
	netShathe = NetworkShathe.NetworkShathe()
	outputs = netShathe.net(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
	output = outputs[len(outputs)-1]
	'''
	# Network
	output = Network.MiniNet(input_x=x, n_classes=n_classes, training=training_flag)
	#output = Network.small(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
	#output = Network.build_fc_densenet(x, n_classes, preset_model='FC-DenseNet103')
	#output=Network.build_mobile_unet(x, preset_model="MobileUNet", num_classes=n_classes)
	#output = erfnetB(x, n_classes,  is_training=training_flag)	

	#output, _ = build_pspnet(x, label_size=[height, width], num_classes=n_classes, preset_model="PSPNet-Res50", pooling_type = "AVG", weight_decay=1e-5, upscaling_method="conv", is_training=training_flag)

	tf.profiler.profile(
		tf.get_default_graph(),
		options=tf.profiler.ProfileOptionBuilder.float_operation())

	#SI ES V4

	print(output.get_shape())
	#tiramisu = Network.DenseTiramisu(16, [4,5,7,10,12,15], n_classes)
	#print(tiramisu.model(x, training_flag))

	# MODEL - ERFNet, with Paszke class weighting


	# Count parameters
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1

		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")



	#output=output[len(output)-1]

	shape_output = output.get_shape()
	label_shape = label.get_shape()

	predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
	labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
	output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
	mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])

	path= '/media/msrobot/discoGordo/Download_april/machine_printed_legible/images/test/COCO_train2014_000000145411.jpg'
	path_labels= '/media/msrobot/discoGordo/Download_april/machine_printed_legible/labels/test/COCO_train2014_000000145411.png'
	# COCO_train2014_000000145411.jpg
	# COCO_train2014_000000037857.jpg
	# COCO_train2014_000000148703.jpg
	img = cv2.imread(path)
	label = cv2.imread(path_labels, 0)
	img = cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA)
	img = preprocess(img)
	img = np.reshape(img, (1, height, width, channels))

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		
		ckpt = tf.train.get_checkpoint_state('./model')  # './model/best'
		ckpt_best = tf.train.get_checkpoint_state('./model/best')  # './model/best'
		if ckpt_best and tf.train.checkpoint_exists(ckpt_best.model_checkpoint_path):
			pass
			#saver.restore(sess, ckpt.model_checkpoint_path)
		

		import time
		image_salida = output_image.eval(feed_dict={x: img, training_flag : False})

		#saver.restore(sess, ckpt_best.model_checkpoint_path)

		first = time.time()
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		image_salida2 = output_image.eval(feed_dict={x: img, training_flag : False})
		second = time.time()
		print(str((second - first)/8.0) + " seconds to load")
		
		'''
		image_salida  = np.reshape(image_salida, (width, height, 1))
		img = cv2.imread(path)
		image_salida = cv2.resize(image_salida, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
		cv2.imshow('image',img)
		cv2.imshow('pred',image_salida)
		cv2.imshow('label',label*255)
		mask_last = img.copy()
		mask_last[:,:,0] = mask_last[:,:,0] * image_salida
		mask_last[:,:,1] = mask_last[:,:,1] * image_salida
		mask_last[:,:,2] = mask_last[:,:,2] * image_salida



		cv2.imshow('last',mask_last)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''