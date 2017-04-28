from six import iteritems
import numpy as np
import random
import sample

def singleton(idx, length):
	return np.matrix([[1 if i == idx else 0 for i in range(length)]])

def load_data():
	samples = sample.get_all_samples(33)
	inputs = [(n, e) for (n, (f, data)) in iteritems(samples) for e in data]
	names = [n for (n, d) in iteritems(samples)]
	out, inp = sample.unzip(inputs)
	out = [singleton(names.index(n), len(names)) for n in out]
	return inp, out, inp, out