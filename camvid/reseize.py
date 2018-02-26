
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import cv2

# tambien las rgb y ademas solo dos labels
max_label = 11
threshold = int(max_label/2)
for filename in glob.glob('./images/*.*'):
	
	print(filename)

	img = cv2.imread(filename)
	img=cv2.resize(img,(224,224))#interpolation=cv2.INTER_NEAREST
	#img[img < threshold] = 0
	#img[img >= threshold] = 1

	# print(np.max(img))



	#img= cv2.imread(filename)
	#img=cv2.resize(img,(224,224))

	cv2.imwrite(filename, img)	

	'''

	img = cv2.imread(filename)
	img=img*20
	cv2.imwrite( filename , img )
'''
