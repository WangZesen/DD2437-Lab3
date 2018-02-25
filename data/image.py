import matplotlib.pyplot as plt
import numpy as np

def show_pattern(pattern, iter = None):
	assert isinstance(pattern, np.ndarray)
	try:
		img = np.reshape(pattern, (32, 32)).tolist()
		plt.imshow(img)
		plt.show()
	except:
		print (pattern)

def show_plot(x, y, x_label = None, y_label = None, title = None):
	assert len(x) == len(y)
	if x_label != None:
		plt.xlabel(x_label)
	if y_label != None:
		plt.ylabel(y_label)
	if title != None:
		plt.title(title)
	plt.plot(x, y)
	plt.show()