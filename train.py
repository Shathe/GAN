import tensorflow as tf
import random
import os
import argparse
import time
from utils.utils import get_parameters
from Loader import Loader
import nets.Network as Network
import math
import numpy as np
import sys
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.keras import backend as K

	
def pre_preprocess (x):
	#return x.astype(np.float32)/127.5 - 1
	return tf.keras.applications.resnet50.preprocess_input(x )


random.seed(os.urandom(7))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train",
					default='./Datasets/camvid')
parser.add_argument("--augmentation", help="Image augmentation", default='segmentation')
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-4)  # 2e-3
parser.add_argument("--min_lr", help="Initial learning rate", default=6e-6)  # 5e-5
parser.add_argument("--batch_size", help="batch_size", default=2)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=100)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="save_model", default=1)
parser.add_argument("--checkpoint_path", help="checkpoint path", default='./models/camvid/')
parser.add_argument("--train", help="if true, train, if not, test", default=1)
args = parser.parse_args()

# Hyperparameter
init_learning_rate = float(args.init_lr)
power_lr = 0.9
min_learning_rate = float(args.min_lr)
save_model = bool(int(args.save_model))
train_or_test = bool(int(args.train))
batch_size = int(args.batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
height = int(args.height)
channels = 3
checkpoint_path = args.checkpoint_path
augmenter = args.augmentation



loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType='segmentation', width=width, height=height, median_frequency=0)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)

# Placeholders
training_flag = tf.placeholder(tf.bool)
input_x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')
label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes], name='output')  
mask_label = tf.placeholder(tf.float32, shape=[None, height, width], name='mask')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Network
output = Network.encoder_decoder_example3(input_x=input_x, n_classes=n_classes, training=training_flag)

# Get shapes
shape_output = tf.shape(output)
label_shape = tf.shape(label)
mask_label_shape = tf.shape(mask_label)

#change shape
predictions = tf.reshape(output, [shape_output[1] * shape_output[2] * shape_output[0], shape_output[3]])
labels = tf.reshape(label, [label_shape[2] * label_shape[1] * label_shape[0], label_shape[3]])
mask_labels = tf.reshape(mask_label, [mask_label_shape[1] * mask_label_shape[0] * mask_label_shape[2]])

# Cross entropy loss
cost = tf.losses.softmax_cross_entropy(labels, predictions, weights=mask_labels)

# Metrics
labels = tf.argmax(labels, 1)
predictions = tf.argmax(predictions, 1)


indices = tf.squeeze(tf.where(tf.greater(mask_labels, 0)))  # ignore labels
labels = tf.cast(tf.gather(labels, indices), tf.int64)
predictions = tf.gather(predictions, indices)

correct_prediction = tf.cast(tf.equal(labels, predictions), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
acc, acc_op = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
iou, conf_mat = tf.metrics.mean_iou(labels, predictions, n_classes)
conf_matrix_all = tf.confusion_matrix(labels, predictions, num_classes=n_classes)


# Different variables
restore_variables = [var for var in tf.global_variables()]  # Change name  if 'logits/semantic' not in var.name and 'decoder' not in var.name
train_variables = [var for var in tf.global_variables()]  # [var for var in tf.trainable_variables() if 'up' in var.name or 'm8' in var.name]
stream_vars = [i for i in tf.local_variables() if 'count' in i.name or 'confusion_matrix' in i.name or 'total' in i.name]

# Count parameters
get_parameters()

K._GRAPH_LEARNING_PHASES[tf.get_default_graph()] = training_flag
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # adamOptimizer does not need lr decay
train = optimizer.minimize(cost, var_list=train_variables)  # VARIABLES TO OPTIMIZE

# For batch norm tensorflow
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
	updates = tf.group(*update_ops)
	cost = control_flow_ops.with_dependencies([updates], cost)

saver = tf.train.Saver(tf.global_variables())
restorer = tf.train.Saver(restore_variables)

if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# get checkpoint if there is one
	ckpt = tf.train.get_checkpoint_state(checkpoint_path)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Loading model...')
		restorer.restore(sess, ckpt.model_checkpoint_path)
		print('Model loaded')

	if train_or_test:

		# Start variables
		best_iou = float('-Inf')
		# EPOCH  loop
		for epoch in range(total_epochs):
			K.set_learning_phase(1)
			# Calculate tvariables for the batch and inizialize others
			time_first = time.time()
			epoch_learning_rate = (init_learning_rate - min_learning_rate) * math.pow(1 - epoch / 1. / total_epochs,
																					  power_lr) + min_learning_rate

			print ("epoch " + str(epoch + 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size))

			total_steps = int(training_samples / batch_size) + 1

			# steps in every epoch
			for step in range(total_steps):

				# get training data
				batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
				train_feed_dict = {
					input_x: pre_preprocess(batch_x),
					label: batch_y,
					learning_rate: epoch_learning_rate,
					mask_label: batch_mask,
					training_flag: True
				}
				
				_, loss = sess.run([train, cost], feed_dict=train_feed_dict)
				# show info
				if step % 10 == 0:
					print 'training loss: ' + str(loss)


			# TEST
			loss_acum = 0.0
			K.set_learning_phase(0)
			for i in xrange(0, testing_samples):
				x_test, y_test, mask_test = loader.get_batch(size=1, train=False)
				test_feed_dict = {
					input_x: pre_preprocess(x_test),
					label: y_test,
					mask_label: mask_test,
					learning_rate: 0,
					training_flag: False
				}
				acc_update, miou_update, mean_acc_update, val_loss = sess.run([acc_op, conf_mat, mean_acc_op, cost],
																			  feed_dict=test_feed_dict)
				acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
																			  feed_dict=test_feed_dict)
				loss_acum = loss_acum + val_loss

			print("TEST")
			print("Accuracy: " + str(acc_total))
			print("miou: " + str(miou_total))
			print("mean accuracy: " + str(mean_acc_total))
			print("loss: " + str(loss_acum / testing_samples))
			#print('matrix_conf')
			#print(matrix_conf)

			sess.run(tf.variables_initializer(stream_vars))

			# save models
			if save_model and best_iou < miou_total:
				best_iou = miou_total
				saver.save(sess=sess, save_path=checkpoint_path + 'model.ckpt')
		   
			loader.suffle_segmentation()

			# show tiem to finish training
			time_second = time.time()
			epochs_left = total_epochs - epoch - 1
			segundos_per_epoch = time_second - time_first
			print(str(segundos_per_epoch * epochs_left) + ' seconds to end the training. Hours: ' + str(
				segundos_per_epoch * epochs_left / 3600.0))



	else:

		# TEST
		loss_acum = 0.0
		K.set_learning_phase(False)

		for i in xrange(0, testing_samples):
			x_test, y_test, mask_test = loader.get_batch(size=1, train=False)
			test_feed_dict = {
				input_x: pre_preprocess(x_test),
				label: y_test,
				mask_label: mask_test,
				learning_rate: 0,
				training_flag: False
			}
			acc_update, miou_update, mean_acc_update, val_loss = sess.run([acc_op, conf_mat, mean_acc_op, cost],
																		  feed_dict=test_feed_dict)
			acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
																		  feed_dict=test_feed_dict)
			loss_acum = loss_acum + val_loss

		print("TEST")
		print("Accuracy: " + str(acc_total))
		print("miou: " + str(miou_total))
		print("mean accuracy: " + str(mean_acc_total))
		print("loss: " + str(loss_acum / testing_samples))
		print('matrix_conf')
		print(matrix_conf)