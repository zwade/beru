from __future__ import print_function
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import random
import signal
import sys
from threading import Timer
import subprocess
from six import iteritems

SELECTION_THRESHOLD = .2
NON_SELECTION_THRESHOLD = .1
SCREEN_UPDATE_RATE = .2
PROGRESS_CHAR = u"\u2593"
NON_PROGRESS_CHAR = u"\u2591"

class TDNN:
	def __init__(self, input_file = None, layers = None, learning_rate = 0.1):
		self.sigmoid = np.vectorize(sspec.expit)
		self.square = np.vectorize(lambda x: x*x)
		self.last = None
		self.total_iterations = 1
		self.current_iteration = 1
		self.total_steps = 1
		self.current_step = 1
		self.error = 0
		self.outcomes = [0, 0, 0, 0, 0]
		self.tests = 0
		self.by_category = None
		self.testing_error = 0
		self.testing_outcomes = [0, 0, 0, 0, 0]
		self.testing_tests = 0
		self.testing_total_steps = 1
		self.testing_current_step = 1
		self.testing_by_category = None
		self.testing = False
		self.timer = None

		if input_file is not None:
			print("\033[36mLoading weights from file...  \033[0m", end="")
			archive = np.load(input_file)
			parts = [archive[file] for file in sorted(archive.files)]
			self.layers = parts[0]
			self.matrices = parts[1:]
			print("\033[36m....Done.\033[0m")
		elif layers is not None:
			print("\033[36mGenerating new weights for layers", layers, "...  \033[0m", end="")
			self.layers = layers
			self.matrices = [np.random.rand(layers[i][0] * (layers[i][1] - layers[i+1][1] + 1) + 1, layers[i+1][0]) / 10 - 0.05 for i in range(len(layers) - 1)]
			print("\033[36m...Done.\033[0m", end="")
		else:
			raise Error("\033[31minput_file or layers must be specified\033[0m")

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

	def train(self, X, Y, Z, iterations = 1000):
		self.testing = False
		self.total_iterations = iterations

		for t in range(iterations):
			error = 0
			outcomes = [0, 0, 0, 0, 0]
			tests = 0
			by_category = dict()

			self.current_iteration = t
			self.total_steps = len(X)

			for i in range(len(X)):
				self.current_step = i
				if Y[i] is None:
					continue
				label = self.forward_propagate(X[i])
				# print(label, Y[i])
				error += np.sum(self.square(Y[i] - label))

				case_concluded = True
				case_correct = True
				arg_max = 0

				for j in range(label.size):
					if label[0, arg_max] < label[0, j]:
						arg_max = j
					if NON_SELECTION_THRESHOLD < label[0, j] < SELECTION_THRESHOLD:
						case_concluded = False
					elif label[0, j] <= NON_SELECTION_THRESHOLD and Y[i][0, j] == 1:
						case_correct = False
					elif label[0, j] >= SELECTION_THRESHOLD and Y[i][0, j] == 0:
						case_correct = False

				# print(label, Y[i], case_correct, case_concluded)

				if Z[i] not in by_category:
					by_category[Z[i]] = [0, 0, 0, 0, 0]

				if case_concluded and case_correct:
					outcomes[0] += 1
					by_category[Z[i]][0] += 1
				elif not case_concluded:
					if Y[i][0, arg_max] == 1:
						outcomes[1] += 1
						by_category[Z[i]][1] += 1
					else:
						outcomes[2] += 1
						by_category[Z[i]][2] += 1
				else:
					if Y[i][0, arg_max] == 1:
						outcomes[3] += 1
						by_category[Z[i]][3] += 1
					else:
						outcomes[4] += 1
						by_category[Z[i]][4] += 1

				tests += 1
				self.back_propagate(Y[i])

			self.error = error
			self.outcomes = outcomes
			self.tests = tests
			self.by_category = by_category

	def test(self, X, Y, Z):
		self.testing = True
		error = 0
		outcomes = [0, 0, 0, 0, 0]
		tests = 0
		by_category = dict()

		self.testing_total_steps = len(X)
		self.testing_incorrect_sum = np.zeros((1, self.layers[-1, 0]))
		self.testing_incorrect_labels_sum = np.zeros((1, self.layers[-1, 0]))

		for i in range(len(X)):
			self.testing_current_step = i
			if Y[i] is None:
				continue

			label = self.forward_propagate(X[i])
			error += np.sum(self.square(Y[i] - label))

			case_concluded = True
			case_correct = True
			arg_max = 0

			for j in range(label.size):
				if label[0, arg_max] < label[0, j]:
					arg_max = j

				if NON_SELECTION_THRESHOLD < label[0, j] < SELECTION_THRESHOLD:
					case_concluded = False
				elif label[0, j] <= NON_SELECTION_THRESHOLD and Y[i][0, j] == 1:
					case_correct = False
				elif label[0, j] >= SELECTION_THRESHOLD and Y[i][0, j] == 0:
					case_correct = False

			if Z[i] not in by_category:
				by_category[Z[i]] = [0, 0, 0, 0, 0]

			if case_concluded and case_correct:
				outcomes[0] += 1
				by_category[Z[i]][0] += 1
			elif not case_concluded:
				if Y[i][0, arg_max] == 1:
					outcomes[1] += 1
					by_category[Z[i]][1] += 1
				else:
					outcomes[2] += 1
					by_category[Z[i]][2] += 1
			else:
				if Y[i][0, arg_max] == 1:
					outcomes[3] += 1
					by_category[Z[i]][3] += 1
				else:
					outcomes[4] += 1
					by_category[Z[i]][4] += 1

			tests += 1
			self.back_propagate(Y[i])

		self.testing_error = error
		self.testing_outcomes = outcomes
		self.testing_tests = tests
		self.testing_by_category = by_category

	def save(self, output_file):
		print("\033[36mSaving weights to", output_file, "...  \033[30m", end="")
		np.savez(output_file, self.layers, *self.matrices)
		print("\033[36m...Done.\033[30m")
	def append_one(self, v):
		return np.insert(v, v.size, 1)

	def print(self):
		print("\033[2J\033[3J\033[;H\033[0m", end="")
		rows, columns = subprocess.check_output(['stty', 'size']).decode().split()
		width = int(columns)
		print()

		if self.testing:
			print()
			print()
			print() 
			self.progress("Testing Cases", [
				(self.testing_current_step, PROGRESS_CHAR, "\033[34m", None),
				(self.testing_total_steps, NON_PROGRESS_CHAR, "\033[0;37m", None)
			], self.testing_total_steps, width)
			print()
		else:
			self.progress("Training Iterations", [
				(self.current_iteration, PROGRESS_CHAR, "\033[34m", None),
				(self.total_iterations, NON_PROGRESS_CHAR, "\033[0;37m", None)
			], self.total_iterations, width)

			print()

			self.progress("Training Cases", [
				(self.current_step, PROGRESS_CHAR, "\033[34m", None),
				(self.total_steps, NON_PROGRESS_CHAR, "\033[30;37m", None)
			], self.total_steps, width)

			print()

		if self.tests > 0:
			self.progress("Training Outcomes", [
				(self.outcomes[0], PROGRESS_CHAR, "\033[32m", "OK"),
				(self.outcomes[1], PROGRESS_CHAR, "\033[36m", "AL"),
				(self.outcomes[2], PROGRESS_CHAR, "\033[33m", "IN"),
				(self.outcomes[3], PROGRESS_CHAR, "\033[35m", "ST"),
				(self.outcomes[4], PROGRESS_CHAR, "\033[31m", "WA")
			], self.tests, width)

			print()
			print("Training Error: {0:.6f}".format(self.error))

			print()

			for (classname, outcomes) in iteritems(self.by_category):
				self.progress(classname, [
					(outcomes[0], PROGRESS_CHAR, "\033[32m", "OK"),
					(outcomes[1], PROGRESS_CHAR, "\033[36m", "AL"),
					(outcomes[2], PROGRESS_CHAR, "\033[33m", "IN"),
					(outcomes[3], PROGRESS_CHAR, "\033[35m", "ST"),
					(outcomes[4], PROGRESS_CHAR, "\033[31m", "WA")
				], sum(outcomes), width)

			print()
		else:
			print("Training Outcomes", " " * (width - 28), "[no data]")
			print(PROGRESS_CHAR * width)

		print()

		if self.testing_tests > 0:
			self.progress("Testing Outcomes", [
				(self.testing_outcomes[0], PROGRESS_CHAR, "\033[32m", "OK"),
				(self.testing_outcomes[1], PROGRESS_CHAR, "\033[36m", "AL"),
				(self.testing_outcomes[2], PROGRESS_CHAR, "\033[33m", "IN"),
				(self.testing_outcomes[3], PROGRESS_CHAR, "\033[35m", "ST"),
				(self.testing_outcomes[4], PROGRESS_CHAR, "\033[31m", "WA")
			], self.testing_tests, width)

			print()
			print("Testing Error: {0:.6f}".format(self.testing_error))
			print()

			for (classname, outcomes) in iteritems(self.testing_by_category):
				self.progress(classname, [
					(outcomes[0], PROGRESS_CHAR, "\033[32m", "OK"),
					(outcomes[1], PROGRESS_CHAR, "\033[36m", "AL"),
					(outcomes[2], PROGRESS_CHAR, "\033[33m", "IN"),
					(outcomes[3], PROGRESS_CHAR, "\033[35m", "ST"),
					(outcomes[4], PROGRESS_CHAR, "\033[31m", "WA")
				], sum(outcomes), width)

			print()



		self.timer = Timer(SCREEN_UPDATE_RATE, self.print)
		self.timer.start()

	def progress(self, title, parts, total, width):
		total_len = len(str(total))

		label = " / ".join([("{0:" + str(total_len) + "d}" + (" {1}" if name is not None else "")).format(size, name) for size, c, color, name in parts])
		print(title, end="")
		print(" " * (width - len(title) - len(label)), end="")
		print(label)

		remaining = width

		for size, c, color, name in parts[:-1]:
			section_width = int(1.0 * width * size / total)
			remaining -= section_width
			print(color, end="")
			print(c * section_width, end="")
			print("\033[0m", end="")

		_, c, color, __ = parts[-1]
		print(color, end="")
		print(c * remaining, end="")
		print("\033[0m")

	def stop(self):
		self.timer.cancel()
