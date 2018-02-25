import numpy as np
import copy, random

class network:
	def __init__(self, dim, sync = True, data = None, verbose = False):
		self.dim = dim
		self.sync = sync
		self._set_weight = False
		self._verbose = verbose
		self._max_iter = 5
		if data != None:
			assert isinstance(data, np.ndarray)
			self.update_weight(data)

	def update_weight(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[1] == self.dim
		self.w = np.zeros((self.dim, self.dim))
		for i in range(data.shape[0]):
			self.w = self.w + np.dot(data[i:i + 1].T, data[i:i + 1])
		# self.w = self.w / self.dim
		'''
		for i in range(self.dim):
			self.w[i][i] = 0
		'''
		self._set_weight = True
			
		if self._verbose:
			print (self.w)
		

	def update_state(self, init_state):
		assert self._set_weight
		assert init_state.shape == (self.dim,)
		if self.sync:
			return self._sync_update_state(init_state)
		else:
			return self._unsync_update_state(init_state)

	def stationary_point(self, init_state):
 		if np.array_equal(self._sign_list(np.dot(init_state, self.w)), init_state):
 			return True
 		else:
 			return False

	def _sign_scala(self, value):
		if value > 0:
			return 1.
		elif value == 0:
			return 0.
		else:
			return -1.


	def _sign_list(self, state):
		for i in range(self.dim):
			state[i] = self._sign_scala(state[i])
		return state

	def _sync_update_state(self, init_state):
		if self._verbose:
			print ("------------state debug-----------")

		old_state = copy.deepcopy(init_state)
		for i in range(self._max_iter):
			new_state = self._sign_list(np.dot(old_state, self.w))

			if self._verbose:
				print ("[Verbose Iteration {}]".format(i + 1))
				print (np.dot(old_state, self.w))
				print (new_state)

			if np.array_equal(new_state, old_state):
				
				if self._verbose:
					print ("[Debug] Converge in {} Iterations".format(i))
					print ("-------------debug end-------------")

				return i, old_state
			old_state = copy.deepcopy(new_state)
		
		print ("[Error] Can't Converge in {} Iterations".format(self._max_iter))

		if self._verbose:
			print ("-------------debug end-------------")

		return -1, new_state

	def _unsync_update_state(self, init_state):
		old_state = copy.deepcopy(init_state)
		for i in range(self._max_iter):
			new_state = copy.deepcopy(old_state)
			index = random.sample(range(self.dim), self.dim)
			for j in range(self.dim):
				new_state[index[j]] = self._sign_scala(np.dot(new_state, self.w[:, index[j]]))
			if np.array_equal(new_state, old_state):
				return i, old_state
		pass
		print ("[Error] Can't Converge in {} Iterations".format(self._max_iter))
		return -1, new_state