from __future__ import print_function
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import random
import signal
import sys
from threading import Timer
import subprocess

SELECTION_THRESHOLD = .75
NON_SELECTION_THRESHOLD = .25
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
		self.correct = 0
		self.inconclusive = 0
		self.incorrect = 0
		self.tests = 0
		self.testing_error = 0
		self.testing_correct = 0
		self.testing_inconclusive = 0
		self.testing_incorrect = 0
		self.testing_tests = 0
		self.testing_total_steps = 1
		self.testing_current_step = 1
		self.testing_incorrect_sum = None
		self.testing_incorrect_labels_sum = None
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

	def train(self, X, Y, iterations = 1000):
		self.testing = False
		self.total_iterations = iterations

		for t in range(iterations):
			error = 0
			correct = 0
			inconclusive = 0
			incorrect = 0
			tests = 0

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

				for j in range(len(label)):
					if NON_SELECTION_THRESHOLD < label[0, j] < SELECTION_THRESHOLD:
						case_concluded = False
					elif label[0, j] <= NON_SELECTION_THRESHOLD and Y[i][0, j] == 1:
						case_correct = False
					elif label[0, j] >= SELECTION_THRESHOLD and Y[i][0, j] == 0:
						case_correct = False

				if case_concluded and case_correct:
					correct += 1
				elif not case_concluded:
					inconclusive += 1
				else:
					incorrect += 1

				tests += 1
				self.back_propagate(Y[i])

			self.error = error
			self.correct = correct
			self.inconclusive = inconclusive
			self.incorrect = incorrect
			self.tests = tests

	def test(self, X, Y):
		self.testing = True
		error = 0
		correct = 0
		inconclusive = 0
		incorrect = 0
		tests = 0
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

			for j in range(len(label)):
				if NON_SELECTION_THRESHOLD < label[0, j] < SELECTION_THRESHOLD:
					case_concluded = False
				elif label[0, j] <= NON_SELECTION_THRESHOLD and Y[i][0, j] == 1:
					case_correct = False
				elif label[0, j] >= SELECTION_THRESHOLD and Y[i][0, j] == 0:
					case_correct = False

			if case_concluded and case_correct:
				correct += 1
			elif not case_concluded:
				inconclusive += 1
				self.testing_incorrect_sum += Y[i]
				self.testing_incorrect_labels_sum += label
			else:
				incorrect += 1
				self.testing_incorrect_sum += Y[i]
				self.testing_incorrect_labels_sum += label

			tests += 1
			self.back_propagate(Y[i])

		self.testing_error = error
		self.testing_correct = correct
		self.testing_inconclusive = inconclusive
		self.testing_incorrect = incorrect
		self.testing_tests = tests

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
			step_bar_done = int(1.0 * self.testing_current_step / self.testing_total_steps * width)
			step_bar_remaining = width - step_bar_done
			step_len = len(str(self.testing_total_steps))
			step_fmt = "{0:" + str(step_len) + "d} / {1:" + str(step_len) + "d}"
			partial_str = step_fmt.format(self.testing_current_step, self.testing_total_steps)
			print("Cases", " " * (width - 5 - len(partial_str)), partial_str, sep="")
			print("\033[34m", end="")
			print(PROGRESS_CHAR * step_bar_done, end="")
			print("\033[0;37m", end="")
			print(NON_PROGRESS_CHAR * step_bar_remaining)
			print("\033[0m", end="")
		else:
			iter_bar_done = int(1.0 * self.current_iteration / self.total_iterations * width)
			iter_bar_remaining = width - iter_bar_done
			iter_len = len(str(self.total_iterations))
			iter_fmt = "{0:" + str(iter_len) + "d} / {1:" + str(iter_len) + "d}"
			partial_str = iter_fmt.format(self.current_iteration, self.total_iterations)
			print("Training Iterations", " " * (width - 19 - len(partial_str)), partial_str, sep="")
			print("\033[34m", end="")
			print(PROGRESS_CHAR * iter_bar_done, end="")
			print("\033[0;37m", end="")
			print(NON_PROGRESS_CHAR * iter_bar_remaining)
			print("\033[0m", end="")

			print()

			step_bar_done = int(1.0 * self.current_step / self.total_steps * width)
			step_bar_remaining = width - step_bar_done
			step_len = len(str(self.total_steps))
			step_fmt = "{0:" + str(step_len) + "d} / {1:" + str(step_len) + "d}"
			partial_str = step_fmt.format(self.current_step, self.total_steps)
			print("Cases", " " * (width - 5 - len(partial_str)), partial_str, sep="")
			print("\033[34m", end="")
			print(PROGRESS_CHAR * step_bar_done, end="")
			print("\033[0;37m", end="")
			print(NON_PROGRESS_CHAR * step_bar_remaining)
			print("\033[0m", end="")

			print()

			if self.tests > 0:
				cases_len = len(str(self.tests))
				outcomes_fmt = "{0:" + str(cases_len) + "d} OK / {1:" + str(cases_len) + "d} ?? / {2:" + str(cases_len) + "d} WA"
				partial_str = outcomes_fmt.format(self.correct, self.inconclusive, self.incorrect)
				print("Training Outcomes", " " * (width - 17 - len(partial_str)), partial_str, sep="")

				training_bar_correct = int(1.0 * self.correct / self.tests * width)
				training_bar_inconclusive = int(1.0 * self.inconclusive / self.tests * width)
				training_bar_incorrect = width - training_bar_correct - training_bar_inconclusive

				print("\033[32m", end="")
				print(PROGRESS_CHAR * training_bar_correct, end="")
				print("\033[33m", end="")
				print(PROGRESS_CHAR * training_bar_inconclusive, end="")
				print("\033[31m", end="")
				print(PROGRESS_CHAR * training_bar_incorrect)
				print("\033[0m", end="")

				print()
				print("Training Error: {0:.6f}".format(self.error))
			else:
				print("Training Outcomes", " " * (width - 28), "[no data]")
				print(PROGRESS_CHAR * width)

		print()

		if self.testing_tests > 0:
			cases_len = len(str(self.testing_tests))
			outcomes_fmt = "{0:" + str(cases_len) + "d} OK / {1:" + str(cases_len) + "d} ?? / {2:" + str(cases_len) + "d} WA"
			partial_str = outcomes_fmt.format(self.testing_correct, self.testing_inconclusive, self.testing_incorrect)
			print("Last Testing Outcomes", " " * (width - 21 - len(partial_str)), partial_str, sep="")

			training_bar_correct = int(1.0 * self.testing_correct / self.testing_tests * width)
			training_bar_inconclusive = int(1.0 * self.testing_inconclusive / self.testing_tests * width)
			training_bar_incorrect = width - training_bar_correct - training_bar_inconclusive

			print("\033[32m", end="")
			print(PROGRESS_CHAR * training_bar_correct, end="")
			print("\033[33m", end="")
			print(PROGRESS_CHAR * training_bar_inconclusive, end="")
			print("\033[31m", end="")
			print(PROGRESS_CHAR * training_bar_incorrect)
			print("\033[0m", end="")

			print()
			print("Testing Error: {0:.6f}".format(self.testing_error))
			print("Testing Failures:", self.testing_incorrect_sum)
			print("Testing Labels:", self.testing_incorrect_labels_sum)


		self.timer = Timer(SCREEN_UPDATE_RATE, self.print)
		self.timer.start()

	def stop(self):
		self.timer.cancel()
