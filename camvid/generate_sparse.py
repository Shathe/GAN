import glob
import cv2
import numpy as np
import csv
import argparse
import ntpath
import PIL
from PIL import Image
import random
import os.path
import os, os.path
from os import listdir
from os.path import isfile, join
import math




if not os.path.exists('sparse'):
    os.makedirs('sparse')

NUM_LABELS = 350


for filename in glob.glob('labels/*.png'): #imagenes test a crear patches


	img = cv2.imread(filename, 0)
	sparse = cv2.imread(filename, 0)
	sparse[:,:] = 255

	i_size, j_size = img.shape
	
	

	
	init_i = (i_size  - math.sqrt(NUM_LABELS) * math.trunc(i_size/math.sqrt(NUM_LABELS)) -1)/2
	init_j = (j_size  - math.sqrt(NUM_LABELS) * math.trunc(j_size/math.sqrt(NUM_LABELS))-1)/2


	for i in xrange(math.trunc(math.sqrt(NUM_LABELS))+1):
		for j in xrange(math.trunc(math.sqrt(NUM_LABELS))+1):
			i_point = int(i * math.trunc(j_size/math.sqrt(NUM_LABELS)) + init_i)
			j_point= int(j * math.trunc(j_size/math.sqrt(NUM_LABELS))+ init_j)

			sparse[i_point,j_point] = img[i_point,j_point]


	cv2.imwrite('sparse/' + ntpath.basename(filename) ,sparse)
				 

	


print('procentaje sobre inicial')
print(NUM_LABELS/480.0/360.0)
print('uno de cada X pixeles reales, sparse')
print(600.0*600.0/NUM_LABELS)
