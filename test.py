import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import random
import math
import os
import NetworkShathe

import argparse
import time
from Loader import Loader
from imgaug import augmenters as iaa
import imgaug as ia
from augmenters import get_augmenter
import tensorflow.contrib.slim as slim
import Network
import cv2
random.seed(os.urandom(9))
#tensorboard --logdir=train:./logs/train,test:./logs/test/

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset to train", default='./camvid')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-4)
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=4)
parser.add_argument("--max_batch_size", help="batch_size", default=4)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=400)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
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

loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType = 'segmentation', width=width, height=height, ignore_label = ignore_label)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)


# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)

# Placeholder para las imagenes.
x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')
label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes+1], name='output')
mask_label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes], name='mask')
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
output = Network.nasnet(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)


#SI ES V4
#output=output[len(output)-1]

shape_output = output.get_shape()
label_shape = label.get_shape()

predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3] - 1]) # tf.reshape(output, [-1])



 
# Metrics


labels = tf.argmax(labels, 2)
predictions = tf.argmax(predictions, 2)

labels = tf.reshape(labels,[tf.shape(labels)[0] * labels.get_shape()[1].value])
predictions = tf.reshape(predictions,[tf.shape(predictions)[0] * predictions.get_shape()[1].value])
print(tf.where(tf.less_equal(labels, n_classes - 1)).get_shape())

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1))) # ignore all labels >= num_classes 
labels = tf.cast(tf.gather(labels, indices), tf.int32)
predictions = tf.gather(predictions, indices)
# esto tabienen la loss?





acc, acc_op  = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
miou, miou_op = tf.metrics.mean_iou(labels, predictions, n_classes)

# args.dataset
 

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('./model')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./model/best')  # './model/best'
	if ckpt_best and tf.train.checkpoint_exists(ckpt_best.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)

	# TEST
	count = 0
	suma_acc = 0
	for i in xrange(0, testing_samples, max_batch_size):
		if i + max_batch_size > testing_samples:
			max_batch_size = testing_samples - i
		x_test, y_test, mask_test = loader.get_batch(size=max_batch_size, train=False, index=i, validation=True)
		count = count + 1
		test_feed_dict = {
			x: x_test,
			label: y_test,
			training_flag: False,
			mask_label: mask_test
		}
		acc_update, acc_total, miou_update, miou_total,mean_acc_total, mean_acc_update = sess.run([acc_op, acc, miou_op, miou, mean_acc, mean_acc_op], feed_dict=test_feed_dict)

	print("Accuracy: " + str(acc_update))
	print("miou: " + str(miou_total))
	print("mean accuracy: " + str(mean_acc_total))

	x_test, y_test, mask_test = loader.get_batch(size=1, train=False, index=0, validation=True)

	import time
	first = time.time()
	predictions = sess.run(output_image, feed_dict={x: x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")


	first = time.time()
	output_image.eval(feed_dict={x: x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")

# mejor complex sin regularizer y droput: 0.80/0.77, 0.44, 0.66
