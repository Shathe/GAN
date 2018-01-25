import os
import numpy as np
from os import walk
from keras.utils.np_utils import to_categorical
import glob
import ntpath
import cv2
import random
from imgaug import augmenters as iaa
import imgaug as ia
from augmenters import get_augmenter

problemTypes=['classification', 'GAN', 'segmentation', 'DVS']

class Loader:
	#poder aumentar datos.. poder batch.. opcion sampleado aleatorio
	#guardar lista de imagenes test y train de una carpeta
	# opcion de que devuelva la mascara de lo que se aumenta
	# opcion del tipo de entrenamiento. Clafiicacion, semantica, gan.. eventos
	def __init__(self, dataFolderPath, width=224, height=224, dim=3, n_classes=10  problemType='classification'):
		self.height = height
		self.width = width
		self.dim = dim 

		# Load filepaths
		files = []
		for (dirpath, dirnames, filenames) in walk(dataFolderPath):
			filenames = [os.path.join(dirpath, filename) for filename in filenames]
			files.extend(filenames)

		self.test_list = [file for file in files if '/test/' in file]
		self.train_list = [file for file in files if '/train/' in file]
		print('Loaded '+ str(len(self.train_list)) +' training samples')
		print('Loaded '+ str(len(self.test_list)) +' testing samples')

		# Check problem type
		if problemType in problemTypes:
			self.problemType=problemType
		else:
			raise Exception('Not valid problemType')


		if problemType == 'classification' or problemType == 'GAN':
			#Extract dictionary to map class -> label
			classes_train = [file.split('/train/')[1].split('/')[0] for file in self.train_list]
			classes_test = [file.split('/test/')[1].split('/')[0] for file in self.test_list]
			classes = np.unique(np.concatenate((classes_train, classes_test)))
			self.classes = {}
			for label in range(len(classes)):
				self.classes[classes[label]] = label

		elif problemType == 'segmentation':
			# The structure has to be dataset/train/images/image.png
			# The structure has to be dataset/train/labels/label.png
			# Separate image and label lists
			self.image_train_list = [file for file in self.train_list if '/images/' in file]
			self.image_test_list = [file for file in self.test_list if '/images/' in file]
			self.label_train_list = [file for file in self.train_list if '/labels/' in file]
			self.label_test_list = [file for file in self.test_list if '/labels/' in file]
			print(self.image_train_list)
			print(self.label_train_list)
			self.n_classes=n_classes

		elif problemType == 'DVS':
			# Yet to know how to manage this data
			pass


	# Returns a random batch of segmentation images: X, Y, mask
	def _get_batch_segmentation(self, size=32, train=True):

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size, self.height, self.width], dtype=np.uint8)
		mask = np.ones([size, self.height, self.width], dtype=np.uint8)

		image_list = self.image_test_list
		label_list = self.label_test_list
		folder = '/test/'
		if train:
			image_list = self.image_train_list
			label_list = self.label_train_list
			folder = '/train/'

		# Get [size] random numbers
		random_numbers = [random.randint(0,len(image_list) - 1) for file in range(size)]
		random_images = [image_list[number] for number in random_numbers]
		random_labels = [label_list[number] for number in random_numbers]

					

		# for every random image, get the image, label and mask.
		# the augmentation has to be done separately due to augmentation
		for index in range(size):
			seq_image, seq_label, seq_mask = get_augmenter(name='segmentation')

			img = cv2.imread(random_images[index])
			label = cv2.imread(random_labels[index],0)
			if img.shape[1] != self.width and img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)
			if label.shape[1] != self.width and label.shape[0] != self.height:
				label = cv2.resize(label, (self.width, self.height), interpolation = cv2.NEAREST)
			macara = mask[index, :, :] 

			  
			#Reshapes for the AUGMENTER framework
			img=img.reshape(sum(((1,),img.shape),()))
			img = seq_image.augment_images(img)  
			label=label.reshape(sum(((1,),label.shape),()))
			label = seq_label.augment_images(label)
			macara=macara.reshape(sum(((1,),macara.shape),()))
			macara = seq_mask.augment_images(macara)
			macara=macara.reshape(macara.shape[1:])
			label=label.reshape(label.shape[1:])
			img=img.reshape(img.shape[1:])


			x[index, :, :, :] = img
			y[index, :, :] = label
			mask[index, :, :] = macara

		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		a, b, c =y.shape
		y = y.reshape((a*b*c))
		y = to_categorical(y, num_classes=self.n_classes)
		y = y.reshape((a,b,c,self.n_classes))
		x = x.astype(np.float32) / 255.0 - 0.5

		return x, y, mask


	# Returns a random batch
	def _get_batch_rgb(self, size=32, train=True):
		augmenter_seq = get_augmenter(name='rgb')

		x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
		y = np.zeros([size], dtype=np.uint8)

		file_list = self.test_list
		folder = '/test/'
		if train:
			file_list = self.train_list
			folder = '/train/'

		# Get [size] random numbers
		random_files = [file_list[random.randint(0,len(file_list) - 1)] for file in range(size)]
		classes = [self.classes[file.split(folder)[1].split('/')[0]] for file in random_files]


		for index in range(size):
			img = cv2.imread(random_files[index])
			if img.shape[1] != self.width and img.shape[0] != self.height:
				img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)

			x[index, :, :, :] = img
			y[index] = classes[index]

		# the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
		y = to_categorical(y, num_classes=len(self.classes))
		# augmentation
		x = augmenter_seq.augment_images(x)
		x = x.astype(np.float32) / 255.0 - 0.5

		return x, y

	# Returns a random batch
	def _get_batch_GAN(self, size=32, train=True):
		return self._get_batch_rgb(size=size, train=train)


	# Returns a random batch
	def _get_batch_DVS(self, size=32, train=True):
		# Yet to know how to manage this data
		pass


	# Returns a random batch
	def get_batch(self, size=32, train=True):
		if self.problemType == 'classification':
			return self._get_batch_rgb(size=size, train=train)
		elif self.problemType == 'GAN':
			return self._get_batch_GAN(size=size, train=train)
		elif self.problemType == 'segmentation':
			return self._get_batch_segmentation(size=size, train=train)
		elif self.problemType == 'DVS':
			return self._get_batch_DVS(size=size, train=train)


if __name__ == "__main__":
	loader = Loader('./dataset_rgb')
	print(loader.classes)
	x, y =loader.get_batch(size=2)
	print(y)
	loader = Loader('./dataset_segmentation', problemType = 'segmentation')
	x, y, mask =loader.get_batch(size=3)
 