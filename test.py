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
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-3)
parser.add_argument("--min_lr", help="Initial learning rate", default=3e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=2)
parser.add_argument("--max_batch_size", help="batch_size", default=2)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=2)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="save_model", default=1)
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

output = Network.complex(input_x=batch_images, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
shape_output = output.get_shape()
label_shape = label.get_shape()


predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])


uniques, idx = tf.unique(predictions)

# funcion de coste: cross entropy (se pued modificar. mediado por todos los ejemplos)
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions))
#cost = -tf.reduce_mean(labels*tf.log(tf.nn.softmax(predctions)), axis=1)

'''
cost1 = tf.reduce_mean(labels*tf.log(tf.nn.softmax(predictions)), axis=1)
weights = np.array([5,5,5,5,5,5,5,5,5,5,5])
cost2 = tf.reduce_mean(cost1*weights, axis=1) 
cost = -tf.reduce_mean(cost2, axis=0)
'''
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)



update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

	# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train = optimizer.minimize(cost) # VARIABLES TO PTIMIZE 


# Accuracy es:

correct_prediction = tf.equal(tf.argmax(labels, 2), tf.argmax(predictions, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc, acc_op  = tf.metrics.accuracy(tf.argmax(labels, 2),tf.argmax(predictions, 2))
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(labels, 2), tf.argmax(predictions, 2), n_classes)
miou, miou_op = tf.metrics.mean_iou(tf.argmax(labels, 2), tf.argmax(predictions, 2), n_classes)


# hacer solo primero para summary


# PONER MASCARA AL ACCYRACY (IGUAL OTRA OPCION DE VALIDATION EN VEZ DE TEST Y QUE DEVUELVE LA CLASE A IGNORARAR PARA LOS WIETHS DE LAS METRICAS)
 

times_show_per_epoch = 15
saver = tf.train.Saver(tf.global_variables())
'''
# initialize the network
init = tf.global_variables_initializer()
'''

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('./model_complex')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./model_complex/best')  # './model/best'
	if ckpt_best and tf.train.checkpoint_exists(ckpt_best.model_checkpoint_path):
		saver.restore(sess, ckpt_best.model_checkpoint_path)
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
			training_flag: False
		}
		accuracy_rates, acc_update, acc_total, miou_update, miou_total,mean_acc_total, mean_acc_update = sess.run([accuracy, acc_op, acc, miou_op, miou, mean_acc, mean_acc_op], feed_dict=test_feed_dict)
		suma_acc = suma_acc + accuracy_rates*max_batch_size


	print("Accuracy: " + str(suma_acc/testing_samples))
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


	
'''
Mejores resultados simple
Accuracy: 0.69807445
miou: 0.35044846
mean accuracy: 0.43492416

Mejores resultados complex
Accuracy: 0.80801904
miou: 0.40060046
mean accuracy: 0.4919342



Tiramisu entrenada de antes acc 88.68, mean acc 48.81, miou 44.36 (con augmnetation)
State-of-the-art  acc 91.5/93  miou 66.9/69.6
'''