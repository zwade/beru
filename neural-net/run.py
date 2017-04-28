from __future__ import print_function
import argparse
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import random
import signal
import sys
from TDNN import TDNN
from load_data import load_data

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

args = parser.parse_args()

print(args)

if args.command == "init":
	net = TDNN(layers = args.layers)
	net.save("nets/" + args.output_file + ".npz")
elif args.command == "train":
	net = TDNN(input_file = "nets/" + args.input_file + ".npz", learning_rate = args.rate)

	def sigint_handler(sig, frame):
		print("Caught SIGINT.")
		net.stop()
		net.save("nets/" + args.output_file + ".npz")
		sys.exit(0)

	signal.signal(signal.SIGINT, sigint_handler)

	print("Loading data...")
	training_data, training_labels, testing_data, testing_labels = load_data()
	print("Done.")

	net.print()

	if args.testfirst:
		net.test(testing_data, testing_labels)

	training_round = 0

	while training_round == 0 or args.loop:
		net.train(training_data, training_labels, args.iterations)
		testing_error = net.test(testing_data, testing_labels)
		training_round += 1

	net.stop()

	print("Training completed.  Saving net...")
	net.save("nets/" + args.output_file + ".npz")