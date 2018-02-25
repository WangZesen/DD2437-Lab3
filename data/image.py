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
