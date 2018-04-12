import random
import numpy as np
from collections import deque


class Replay_Buffer(object):
	def __init__(self, buffer_size=5000, batch_size=32):
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.buffer = deque([], maxlen=self.buffer_size)

	def add(self, s, a, r, s2, done):
		exp = (s, a, r, s2, done)
		self.buffer.append(exp)

	def size(self):
		return len(self.buffer)

	def sample_batch(self):
		batch = []

		if len(self.buffer) < self.batch_size:
			batch = random.sample(list(self.buffer), len(self.buffer))
		else:
			batch = random.sample(list(self.buffer), self.batch_size)

		return batch

	def clear_buffer(self):
		self.buffer.clear()

