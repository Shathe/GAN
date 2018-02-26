import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import random
import math
import os
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
parser.add_argument("--tensorboard", help="Monitor with Tensorboard", default=0)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=1e-3)
parser.add_argument("--min_lr", help="Initial learning rate", default=5e-8)
parser.add_argument("--init_batch_size", help="batch_size", default=4)
parser.add_argument("--max_batch_size", help="batch_size", default=12)
parser.add_argument("--n_classes", help="number of classes to classify", default=21)
parser.add_argument("--ignore_label", help="class to ignore", default=255)
parser.add_argument("--epochs", help="Number of epochs to train", default=20)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="dropout_rate", default=0)
args = parser.parse_args()



# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model ))
tensorboard = bool(int(args.tensorboard))
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
x, y, mask =loader.get_batch(size=3)
print(x.shape)
print(y.shape)
print(mask.shape)

# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)

# Placeholder para las imagenes.
x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')
batch_images = tf.reshape(x, [-1, height, width, channels])
batch_images = tf.reverse(batch_images, axis=[-1]) #opencv rgb -bgr

label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes], name='output')
batch_labels = tf.reshape(label, [-1, height, width, n_classes])
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

# Para poder modificarlo
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

output = Network.simple(input_x=batch_images, n_classes=n_classes, width=width, height=height, channels=channels)
shape_output = output.get_shape()
label_shape = label.get_shape()


predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])


uniques, idx = tf.unique(predictions)

# funcion de coste: cross entropy (se pued modificar. mediado por todos los ejemplos)
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions))
#cost = -tf.reduce_mean(labels*tf.log(tf.nn.softmax(predctions)), axis=1)








# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost) # VARIABLES TO PTIMIZE 


# Accuracy es:
correct_prediction = tf.equal(tf.argmax(predictions, 2), tf.argmax(labels, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = tf.metrics.accuracy(labels, predictions)
prec = tf.metrics.precision(labels,predictions)
miou = tf.metrics.mean_iou(labels, predictions, n_classes)


# hacer solo primero para summary




# Scalar summaries always
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
'''
tf.summary.scalar('acc_total', acc[0])
tf.summary.scalar('acc_update', acc[1])
tf.summary.scalar('prec_total', prec[0])
tf.summary.scalar('prec_update', prec[1])
tf.summary.scalar('miou_total', miou[0])
tf.summary.scalar('miou_update', miou[1])
tf.summary.scalar('learning_rate', learning_rate)
'''



if int(args.dimensions) == 3:

	tf.summary.image('input', batch_images, max_outputs=10)
	#tf.summary.image('input_re', tf.cast((batch_images + 0.5) * 255, tf.uint8), max_outputs=10)

else:
	tf.summary.image('input_0-3', batch_images[:, :, :, 0:3], max_outputs=10)

output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
tf.summary.image('output', output_image, max_outputs=10)
label_image = tf.expand_dims(tf.cast(tf.argmax(label, 3), tf.float32), -1)
tf.summary.image('label', label_image, max_outputs=10)



total_parameters = 0
for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape()
	variable_parameters = 1
	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters
print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")



times_show_per_epoch = 5
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state('./model')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./model/best')  # './model/best'
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())


	merged = tf.summary.merge_all()
	writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
	writer_test = tf.summary.FileWriter('./logs/test', sess.graph)

	global_step = 0
	epoch_learning_rate = init_learning_rate
	batch_size_decimal = float(init_batch_size)
	best_val_loss = float('Inf')
	# EPOCHS
	for epoch in range(total_epochs):
		time_first=time.time()
		batch_size = int(batch_size_decimal)
		print ("epoch " + str(epoch+ 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size) )
		# Simple learning rate decay
		total_batch = int(training_samples / batch_size)
		show_each_steps = int(total_batch / times_show_per_epoch)

		val_loss_acum = 0
		# steps in every epoch
		for step in range(total_batch):
			batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True)


			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				training_flag: True
			}

			_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

			if step % show_each_steps == 0:
				global_step += show_each_steps
				train_summary, train_accuracy, = sess.run([merged, accuracy], feed_dict=train_feed_dict)
				#train_summary, train_accuracy, acc_total, acc_update, prec_total, prec_update, miou_total, miou_update = sess.run([merged, accuracy, acc[0], acc[1], prec[0], prec[1], miou[0], miou[1]], feed_dict=train_feed_dict)
				print("TRAIN")

				print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
				'''
				print("Step:", step, "acc_total:", acc_total, "acc_update:", acc_update)
				print("Step:", step, "prec_update:", prec_update, "prec_total:", prec_total)
				print("Step:", step, "miou_update:", miou_update, "miou_total:", miou_total)
				'''
				writer_train.add_summary(train_summary, global_step=global_step/show_each_steps)


				batch_x_test, batch_y_test, batch_mask = loader.get_batch(size=batch_size, train=False)



				test_feed_dict = {
					x: batch_x_test,
					label: batch_y_test,
					learning_rate: epoch_learning_rate,
					training_flag: False
				}
				print("TEST")

				test_summary, accuracy_rates, val_loss= sess.run([merged, accuracy, cost], feed_dict=test_feed_dict)
				#test_summary, accuracy_rates, val_loss, acc_total, acc_update, prec_total, prec_update, miou_total, miou_update = sess.run([merged, accuracy, cost, acc[0], acc[1], prec[0], prec[1], miou[0], miou[1]], feed_dict=test_feed_dict)
				print("Step:", step, "Loss:", val_loss, "Testing accuracy:", accuracy_rates)
				'''
				print("Step:", step, "acc_total:", acc_total, "acc_update:", acc_update)
				print("Step:", step, "prec_update:", prec_update, "prec_total:", prec_total)
				print("Step:", step, "miou_update:", miou_update, "miou_total:", miou_total)
				'''
				writer_test.add_summary(test_summary, global_step=global_step/show_each_steps)
				val_loss_acum = val_loss_acum + val_loss

		print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates, '/ val_loss =', val_loss_acum)
		if save_model:
			print(save_model)
			saver.save(sess=sess, save_path='./model/dense.ckpt')
		if save_model and best_val_loss > val_loss_acum:
			best_val_loss = val_loss_acum
			saver.save(sess=sess, save_path='./model/best/dense.ckpt')

		time_second=time.time()
		epochs_left = total_epochs - epoch - 1
		segundos_per_epoch=time_second-time_first
		print(str(segundos_per_epoch * epochs_left)+' seconds to end the training. Hours: ' + str(segundos_per_epoch * epochs_left/3600.0))
	
		#agument batch_size per epoch and decrease the learning rate
		epoch_learning_rate = init_learning_rate * math.pow(change_lr_epoch, epoch)
		batch_size_decimal = batch_size_decimal + change_batch_size




	'''

	# TEST
	count = 0
	accuracy_sum = 0.00
	for i in xrange(0, testing_samples, batch_size):
		if i + batch_size > testing_samples:
			batch_size = testing_samples - i
		x_test, y_test, mask_test = loader.get_batch(size=batch_size, train=False, index=i)
		count = count + 1
		test_feed_dict = {
			x: x_test,
			label: y_test,
			training_flag: False
		}
		accuracy_rates = sess.run([accuracy], feed_dict=test_feed_dict)
		print(accuracy_rates)
		accuracy_sum = accuracy_sum + accuracy_rates[0]*batch_size

	print("Accuracy total: " + str(accuracy_sum / testing_samples))

	x_test, y_test, mask_test = loader.get_batch(size=1, train=False, index=0)

	import time
	first = time.time()
	predictions = sess.run(output_image, feed_dict={x: batch_x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")

	print(np.unique(predictions))


	first = time.time()
	output_image.eval(feed_dict={x: batch_x_test, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")


	print(np.unique(predictions))
	'''
