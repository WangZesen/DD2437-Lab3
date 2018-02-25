from sys import argv
import numpy as np
from data import *
from net import *

assert len(argv) == 2
problem_label = argv[1]

if problem_label == "2.2":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1])
	network.update_weight(train)
	for i in range(train.shape[0]):
		print ("Stationary Point:", network.stationary_point(train[0]))

if problem_label == "3.1.1":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	for i in range(test.shape[0]):
		num_iter, final_state = network.update_state(test[i])
		print ("[Test case {}]".format(i + 1).rjust(16), test[i])
		print ("[Test result {}]".format(i + 1).rjust(16), final_state)
		print ("[Ground truth {}]".format(i + 1).rjust(16), train[i])

if problem_label == "3.1.2":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	count = 0
	for i in range(256):
		cur_state = data.get_decode_pattern(i)
		if network.stationary_point(cur_state):
			count += 1
			print (cur_state)
	print ("# Attractors:", count)

if problem_label == "3.1.3":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	test = np.array([-1., -1., -1., -1., -1., -1., -1., -1.])
	num_iter, final_state = network.update_state(test)
	print ("[Test case]".rjust(13), test)
	print ("[Test result]".rjust(13), final_state)

if problem_label == "3.2.1":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)