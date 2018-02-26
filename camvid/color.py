import argparse
import PIL
from PIL import Image
import numpy as np
import os.path
import glob
import random
import ntpath
import cv2



for filename in glob.glob('labels_ignore/*.png'):
	print(filename)
	img = cv2.imread(filename)
	print(sum(sum((img == 3))))
	img[img == 255]=255
	cv2.imwrite(filename,img)

