from __future__ import print_function
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import random
import signal
import sys

SELECTION_THRESHOLD = .5

class TDNN:
	def __init__(self, input_file = None, layers = None, learning_rate = 0.1):
		self.sigmoid = np.vectorize(sspec.expit)
		self.square = np.vectorize(lambda x: x*x)
		self.last = None

		if input_file is not None:
			print("Loading weights from file...")
			archive = np.load(input_file)
			parts = [archive[file] for file in archive.files]
			self.layers = parts[0]
			self.matrices = parts[1:]
			print(self.matrices)
		elif layers is not None:
			print("Generating new weights for layers", layers, "...")
			self.layers = layers
			self.matrices = [np.random.rand(layers[i][0] * (layers[i][1] - layers[i+1][1] + 1) + 1, layers[i+1][0]) / 10 - 0.05 for i in range(len(layers) - 1)]
		else:
			raise Error("input_file or layers must be specified")

		self.learning_rate = learning_rate

	def forward_propagate(self, inp):
		values = np.matrix(inp)
		self.last = [values]

		for i in range(len(self.layers) - 1):
			M = self.matrices[i]
			sample_width = self.layers[i, 1] - self.layers[i+1, 1] + 1
			groups = [self.sigmoid(self.append_one(values[0, j * self.layers[i, 0] : (j + sample_width) * self.layers[i, 0]]) * M) for j in range(self.layers[i+1][1])]
			values = np.concatenate(groups, axis=1)
			self.last.append(values)

		return values

	def back_propagate(self, target):
		if self.last is None:
			return

		error = self.last[-1] - target

		for i in range(len(self.matrices) - 1, -1, -1):
			transpose = np.copy(self.matrices[i]).T
			next_error = np.zeros((1, self.layers[i, 0] * self.layers[i, 1]))
			offset = np.zeros(self.matrices[i].shape)
			for j in range(self.layers[i+1, 1]):
				# print(self.layers[i+1])
				sample_width = self.layers[i, 1] - self.layers[i+1, 1] + 1
				computed = self.last[i+1][0, j * self.layers[i+1, 0] : (j + 1) * self.layers[i+1, 0]]
				used_input = self.last[i][0, j * self.layers[i, 0] : (j + sample_width) * self.layers[i, 0]]
				computed_error = error[0, j * self.layers[i+1, 0] : (j + 1) * self.layers[i+1, 0]]
				# print("C", computed, computed.size)
				delta = np.multiply(np.multiply(computed, np.ones((1, computed.size)) - computed), computed_error)
				# print("D", delta)
				# print(transpose)
				partial_error = delta * transpose
				partial_error = partial_error[:,:-1]
				# print(next_error)
				# print(partial_error)
				# print(partial_error.shape)
				for k in range(0, partial_error.size):
					next_error[0, j * self.layers[i, 0] + k] += partial_error[0, k]
				# print(self.matrices[i].shape, self.append_one(used_input).T.shape, delta.shape)
				offset += self.append_one(used_input).T * delta * self.learning_rate * -1
			self.matrices[i] += offset / self.layers[i+1, 1]
			error = next_error / self.layers[i+1, 1]

	def train(self, X, Y, iterations = 1000):
		for t in range(iterations):
			error = 0.0
			correct = 0.0
			inconclusive = 0.0
			incorrect = 0.0
			tests = 0
			for i in range(len(X)):
				if i % 100 == 0:
					pass # print("Training: iteration", t, "case", i)
				if Y[i] is None:
					continue
				label = self.forward_propagate(X[i])
				# print(label, Y[i])
				error += np.sum(self.square(Y[i] - label))
				if label[0, np.argmax(Y[i])] > SELECTION_THRESHOLD and max([label[0, k] if k != np.argmax(Y[i]) else 0 for k in range(label.size)]) < SELECTION_THRESHOLD:
					correct += 1
				elif max([label[0, k] for k in range(label.size)]) < SELECTION_THRESHOLD:
					inconclusive += 1
				else:
					incorrect += 1
				self.back_propagate(Y[i])
				tests += 1
			print("Training error: ", error / tests, " (", correct / tests, " correct, ", inconclusive / tests, " inconclusive, ", incorrect / tests, " incorrect)", sep="")
			print(self.matrices)

	def test(self, X, Y):
		error = 0.0
		tests = 0
		correct = 0

		for i in range(len(X)):
			label = self.forward_propagate(X[i])
			error += np.sum(self.square(Y[i] - label))
			print(label, Y[i])
			if label[0, np.argmax(Y[i])] > SELECTION_THRESHOLD and max([label[0, k] if k != np.argmax(Y[i]) else 0 for k in range(label.size)]) < SELECTION_THRESHOLD:
					correct += 1
			tests += 1

		return (error / tests, correct / tests)

	def save(self, output_file):
		print("Saving weights to", output_file, "...")
		np.savez(output_file, self.layers, *self.matrices)

	def append_one(self, v):
		return np.insert(v, v.size, 1)
