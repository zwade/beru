import numpy as np
import random

def load_data():
	# inp = [np.matrix([(x >> i) % 2 for i in range(10)]) for x in range(1024)]
	# out = [np.matrix([0 if "01110" in "{0:010b}".format(x) else 1, 1 if "01110" in "{0:010b}".format(x) else 0]) for x in range(1024)]
	# random.shuffle(inp)
	# random.shuffle(out)
	# return inp, out, inp, out
	inp = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
	out = inp
	return inp, out, inp, out