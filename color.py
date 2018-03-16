import argparse
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import ntpath
import cv2




for filename in glob.glob( 'labels/*/*.png'):
	print(filename)

	img = cv2.imread(filename)
	img[img == 2]=0
	img[img == 3]=2
	img[img == 4]=1
	img[img == 5]=0
	img[img == 6]=0
	img[img == 7]=0
	img[img == 8]=1
	img[img == 9]=0
	img[img == 10]=2


	cv2.imwrite(filename,img)

