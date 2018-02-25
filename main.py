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

elif problem_label == "3.1.1":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	for i in range(test.shape[0]):
		num_iter, final_state = network.update_state(test[i])
		print ("[Test case {}]".format(i + 1).rjust(16), test[i])
		print ("[Test result {}]".format(i + 1).rjust(16), final_state)
		print ("[Ground truth {}]".format(i + 1).rjust(16), train[i])

elif problem_label == "3.1.2":
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

elif problem_label == "3.1.3":
	train, test = data.get_toy_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	test = np.array([-1., -1., -1., -1., -1., -1., -1., -1.])
	num_iter, final_state = network.update_state(test)
	print ("[Test case]".rjust(13), test)
	print ("[Test result]".rjust(13), final_state)

elif problem_label == "3.2": # show all patterns
	train, test = data.get_image_example_data()
	for i in range(train.shape[0]):
		image.show_pattern(train[i])
	for i in range(test.shape[0]):
		image.show_pattern(test[i])

elif problem_label == "3.2.1":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train)
	for i in range(test.shape[0]):
		image.show_pattern(test[i])
		num_iter, final_state = network.update_state(test[i])
		image.show_pattern(final_state)

elif problem_label == "3.2.2":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = False, show_gap = 1, show_handle = image.show_pattern)
	network.update_weight(train)
	for i in range(test.shape[0]):
		num_iter, final_state = network.update_state(test[i])

elif problem_label == "3.3.1":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1])
	network.update_weight(train)
	for i in range(train.shape[0]):
		print ("[Case {}]".format(i), network.get_energy(train[i]))

elif problem_label == "3.3.2":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1])
	network.update_weight(train)
	for i in range(test.shape[0]):
		print ("[Case {}]".format(i), network.get_energy(test[i]))

elif problem_label == "3.3.3":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = False, show_gap = 1, show_handle = image.show_pattern)
	network.update_weight(train)
	for i in range(test.shape[0]):
		print ("[Case {}]".format(i))
		num_iter, final_state = network.update_state(test[i])

elif problem_label == "3.3.4":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = False, show_gap = 50, show_handle = image.show_pattern)
	network.update_weight_normal()
	for i in range(test.shape[0]):
		print ("[Case {}]".format(i))
		num_iter, final_state = network.update_state(test[i])

elif problem_label == "3.3.5":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = False, show_gap = 5, show_handle = image.show_pattern)
	network.update_weight_symmetry()
	for i in range(test.shape[0]):
		print ("[Case {}]".format(i))
		num_iter, final_state = network.update_state(test[i])

elif problem_label == "3.4":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = True)
	network.update_weight(train[0:3])
	exp_time = 10
	x = []
	y = []
	for i in range(50):
		x.append(i * 2)
		count = 0
		for j in range(exp_time):
			for k in range(3):
				noise_train = data.flip_pattern(train[k], int(train.shape[1] / 100 * i * 2))
				num_iter, final_state = network.update_state(noise_train)
				if np.array_equal(final_state, train[k]):
					count += 1
		y.append(count / exp_time / 3)
	image.show_plot(x, y, "noise rate", "recover rate")

elif problem_label == "3.5.1":
	train, test = data.get_image_example_data()
	network = net.network(train.shape[1], sync = True)
	for cap in range(3, train.shape[0]):
		network.update_weight(train[0:cap])
		exp_time = 10
		x = []
		y = []
		for i in range(50):
			x.append(i * 2)
			count = 0
			for j in range(exp_time):
				for k in range(cap):
					noise_train = data.flip_pattern(train[k], int(train.shape[1] / 100 * i * 2))
					num_iter, final_state = network.update_state(noise_train)
					if np.array_equal(final_state, train[k]):
						count += 1
			y.append(count / exp_time / 3)
		image.show_plot(x, y, "noise rate", "recover rate", "store {} patterns".format(cap))

elif problem_label == "3.5.2":
	pass

else:
	print ("Invalid Problem Label!")