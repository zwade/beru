from __future__ import print_function
import argparse
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import random
import sample
import signal
import sys
from TDNN import TDNN
from load_data import load_data
import pyaudio
from array import array
import math
from scipy import fftpack
import subprocess
from time import time

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def tdnn_layer(s):
	try:
		nodes, timeshifts = map(int, s.split(","))
		return nodes, timeshifts
	except:
		raise argparse.ArgumentTypeError("Layer must be specified as nodes,timeshifts")

parser = argparse.ArgumentParser(description="Train and test a TDNN.")
subparsers = parser.add_subparsers(dest="command")

init = subparsers.add_parser("init")
init.add_argument("output_file", metavar = "outfile", help = "Where to store the neural net")
init.add_argument("layers", metavar = "layers", type = tdnn_layer, nargs = "+", help = "The size of each layer")

train = subparsers.add_parser("train")
train.add_argument("input_file", metavar = "infile", help = "Weights file from which to load")
train.add_argument("output_file", metavar = "outfile", help = "Weights file to which to store")
train.add_argument("--iterations", "-i", type = int, help = "The number of training iterations to run per round", default = 1000)
train.add_argument("--loop", "-l", action = "store_true", help = "Train in a loop, running test data after each round of training")
train.add_argument("--rate", "-r", type = float, help = "TDNN learning rate", default = 0.1)
train.add_argument("--testfirst", "-t", action = "store_true", help = "Start with a round of testing")
train.add_argument("--fraction", "-f", type = int, help = "What proportion of the input data is used for training", default = 1)

expand = subparsers.add_parser("expand")
expand.add_argument("input_file", metavar = "infile", help = "Weights file from which to load")
expand.add_argument("output_file", metavar = "outfile", help = "Weights file to which to store")
expand.add_argument("layer", type = int, help = "Which layer to expand")
expand.add_argument("amount", type = int, help = "How much to expand the layer by")

classify = subparsers.add_parser("classify")
classify.add_argument("input_file", metavar = "infile", help = "Weights file from which to load")
classify.add_argument("style", metavar = "style", help = "Which input style to use (frq vs cor)", default = "frq")

args = parser.parse_args()

if args.command == "init":
	net = TDNN(layers = args.layers)
	net.save(dir_path + "/nets/" + args.output_file + ".npz")
elif args.command == "train":
	net = TDNN(input_file = dir_path + "/nets/" + args.input_file + ".npz", learning_rate = args.rate)

	def sigint_handler(sig, frame):
		print("Caught SIGINT.")
		net.stop()
		net.save(dir_path + "/nets/" + args.output_file + ".npz")
		sys.exit(0)

	signal.signal(signal.SIGINT, sigint_handler)

	print("Loading data...")
	tr_X, tr_Y, tr_Z, ts_X, ts_Y, ts_Z, class_names = load_data(args.fraction)
	print("Done.")

	net.class_names = class_names

	net.print()

	if args.testfirst:
		net.test(ts_X, ts_Y, ts_Z)

	training_round = 0

	while training_round == 0 or args.loop:
		net.train(tr_X, tr_Y, tr_Z, args.iterations)
		testing_error = net.test(ts_X, ts_Y, ts_Z)
		training_round += 1

	net.stop()

	print("Training completed.  Saving net...")
	net.save(dir_path + "/nets/" + args.output_file + ".npz")
elif args.command == "expand":
	input_file = dir_path + "/nets/" + args.input_file + ".npz"
	archive = np.load(input_file)
	parts = [archive[file] for file in sorted(archive.files)]

	layers = parts[0]
	matrices = parts[1:]
	layers[args.layer, 0] += args.amount

	if args.layer > 0:
		to_concat = np.random.rand(matrices[args.layer - 1].shape[0], args.amount) / 10 - 0.05
		matrices[args.layer - 1] = np.concatenate([matrices[args.layer - 1], to_concat], 1)

	if args.layer < layers.shape[0] - 1:
		parts = layers[args.layer, 1] - layers[args.layer + 1, 1] + 1
		last_row = np.matrix(matrices[args.layer][-1,:])
		matrix = matrices[args.layer][:-1,:]
		splits = np.split(matrix, parts)
		splits = [np.concatenate([part, np.random.rand(args.amount, matrix.shape[1]) / 10 - 0.05]) for part in splits]
		matrices[args.layer] = np.concatenate([np.concatenate(splits), last_row])

	np.savez(dir_path + "/nets/" + args.output_file + ".npz", layers, *matrices)
elif args.command == "classify":
	FORMAT = pyaudio.paInt16
	MIN_FREQ = 20000
	MAX_FREQ = 40000
	RATE   = 96000
	NUM_FQS = 92
	NUM_TIME = 16
	TEST_RATE = 1
	TIMEOUT = 5
	CHUNK  = 2 * RATE // NUM_TIME
	GESTURES = ["o-cw-right", "x-right", "down-right", "s-right"]
	THRESHOLDS = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3]]
	PROGRESS_CHAR = u"\u2593"
	VERSION = sample.FREQUENCY if args.style.lower() == "frq" else sample.AUTOCORRELATION

	def progress(title, parts, total, width):
		total_len = len(str(total))

		label = " / ".join([("{0:4f}" + (" {1}" if name is not None else "")).format(size, name) for size, c, color, name in parts])
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

	dir_path = os.path.dirname(os.path.realpath(__file__))

	audio_inst = pyaudio.PyAudio()
	stream = audio_inst.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)
	net = TDNN(input_file = dir_path + "/nets/" + args.input_file + ".npz", learning_rate = 0)

	windows = []
	outs = []
	cnt = 0
	countdown = TIMEOUT

	while True:
		data = np.array(array("h", stream.read(CHUNK, exception_on_overflow = False)))
		windows.append(data)

		if len(windows) < NUM_TIME:
			continue
		elif len(windows) > NUM_TIME:
			windows = windows[1:]

		cnt += 1

		if cnt % TEST_RATE != 0:
			continue

		data = np.concatenate(windows)
		s = sample.Sample.from_data(data, RATE)

		current = s.get_subtime_data(NUM_FQS, NUM_TIME, VERSION)

		# print(current)

		out = net.forward_propagate(current, False)

		# print(out)
		outs.append(out)

		if len(outs) > TIMEOUT:
			outs = outs[1:]

		final = np.median(outs, 0)

		print("\033[2J\033[3J\033[;H\033[0m", end="")
		rows, columns = subprocess.check_output(['stty', 'size']).decode().split()
		width = int(columns)

		for i in range(len(GESTURES)):
		# 	progress(GESTURES[i], [
		# 		(final[0, i], PROGRESS_CHAR, "\033[31m", None),
		# 		# (out[0, i] - final[0, i], PROGRESS_CHAR, "\033[34m", None),
		# 		(1 - final[0, i], " ", "", None)
		# 	], 1, width)
			match = True
			for j in range(len(GESTURES)):
				if (out[0, j] > THRESHOLDS[i][j] and j != i) or (out[0, j] < THRESHOLDS[i][j] and j == i):
					match = False

			print("{}{:20s} {:.5f}\033[0m".format("\033[32m" if match else "\033[31m", GESTURES[i] + ":", out[0, i]))

		# if np.max(final) > .5:
		# 	countdown -= 1
		# 	if countdown == 0:
		# 		countdown = TIMEOUT
		# 		selected = np.argmax(final)
		# 		print()
		# 		print("Accepted gesture:", GESTURES[selected])
		# 		input("Press enter to continue")

