
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import cv2


for filename in glob.glob('./labels/*.*'):
	
	print(filename)
	img = Image.open(filename)
	img = img.resize((224, 224), PIL.Image.NEAREST)

	img.save(filename)	

	'''

	img = cv2.imread(filename)
	img=img*20
	cv2.imwrite( filename , img )
'''
