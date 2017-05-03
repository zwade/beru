from six import iteritems
import numpy as np
import random
import sample

NUM_FQS  = 48
NUM_TIME = 20
TIME_LEN = 2.0 / NUM_TIME
GESTURES = ["o-cw-right", "x-right", "down-right", "s-right", "noise"]

def singleton(idx, length):
	return np.matrix([[1 if i == idx else 0 for i in range(length)]])

def load_data(fraction):
	samples = sample.get_all_samples(NUM_FQS, NUM_TIME, fraction)

	inputs_tr = [(n, e[:NUM_FQS*NUM_TIME]) for (n, (f, data)) in iteritems(samples['training']) if n in GESTURES for e in data]
	inputs_ts = [(n, e[:NUM_FQS*NUM_TIME]) for (n, (f, data)) in iteritems(samples['test']) if n in GESTURES for e in data]

	random.shuffle(inputs_tr)
	random.shuffle(inputs_ts)

	gesture_only = lambda n: "-".join(n.split("-")[:-1])

	names = [n for n in GESTURES if n != "noise"]
	print(names)

	out_tr, inp_tr = sample.unzip(inputs_tr)
	out_ts, inp_ts = sample.unzip(inputs_ts)

	class_tr = out_tr
	class_ts = out_ts

	out_tr = [np.zeros((1, len(names))) if n == "noise" else singleton(names.index(n), len(names)) for n in out_tr]
	out_ts = [np.zeros((1, len(names))) if n == "noise" else singleton(names.index(n), len(names)) for n in out_ts]

	return inp_tr, out_tr, class_tr, inp_ts, out_ts, class_ts, GESTURES
