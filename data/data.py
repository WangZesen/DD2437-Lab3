import numpy as np
import copy, random

def get_toy_example_data():
	stored_partterns = [[-1., -1., 1., -1., 1., -1., -1., 1.],
						[-1., -1., -1., -1., -1., 1., -1., -1.],
						[-1., 1., 1., -1., -1., 1., -1., 1.]]
	test_patterns = [[1., -1., 1., -1., 1., -1., -1., 1.],
					[1., 1., -1., -1., -1., 1., -1., -1.],
					[1., 1., 1., -1., 1., 1., -1., 1.]]
	return np.array(stored_partterns), np.array(test_patterns)

def get_image_example_data():
	f = open("data/pict.dat", "r")
	content = f.read().split(",")
	train = np.zeros((9, 1024))
	for i in range(9):
		for j in range(1024):
			train[i][j] = int(content[i * 1024 + j])
	test = np.zeros((2, 1024))
	for i in range(2):
		for j in range(1024):
			test[i][j] = int(content[(i + 9) * 1024 + j])			
	return train, test

def get_random_sample_data(n = 300):
	train = np.zeros((n, 100))
	for i in range(n):
		for j in range(100):
			train[i][j] = random.randint(0, 1) * 2 - 1
	return train
	
def get_decode_pattern(value):
	pattern = np.zeros((8,))
	for i in range(8):
		pattern[i] = 1 if value % 2 else -1
		value = value // 2
	return pattern

def flip_pattern(pattern, n):
	new_pattern = copy.deepcopy(pattern)
	index = random.sample(range(pattern.shape[0]), n)
	for i in range(n):
		new_pattern[index[i]] = -1 * new_pattern[index[i]]
	return new_pattern