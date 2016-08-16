from skimage import data, io, filters, color, measure


import numpy as np
import matplotlib.pyplot as plt


filename = '/Users/Tanmay/Development/Guernica/train_4/4.jpg'

image = io.imread(filename)
print type(image)
print image.shape

hsv = color.rgb2hsv(image)
io.imshow(hsv[:,:,0])
io.show();

# edges = filters.sobel(image[:,:,1])
# # io.imshow(image)
# io.imshow(edges)
# io.show()
