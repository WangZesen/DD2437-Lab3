import matplotlib.pyplot as plt
import numpy as np

def show_pattern(pattern, iter = None):
	assert isinstance(pattern, np.ndarray)
	try:
		img = np.reshape(pattern, (32, 32)).tolist()
		if iter != None:
			plt.set_title("The Result of {} Iteration".format(iter))
		plt.imshow(img)
		plt.show()
	except:
		print ("Error in showing pattern!")
	
