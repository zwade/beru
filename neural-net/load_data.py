import numpy as np
import random
import sample

def singleton(idx, length):
	return [1 if i == idx else 0 for i in range(length)]

def load_data():
	# inp = [np.matrix([(x >> i) % 2 for i in range(10)]) for x in range(1024)]
	# out = [np.matrix([0 if "01110" in "{0:010b}".format(x) else 1, 1 if "01110" in "{0:010b}".format(x) else 0]) for x in range(1024)]
	# random.shuffle(inp)
	# random.shuffle(out)
	# return inp, out, inp, out

        samples = sample.get_all_samples(33)

        inputs_tr = [(n, e) for (n, (f, data)) in samples['training'].iteritems() for e in data]
        inputs_ts = [(n, e) for (n, (f, data)) in samples['training'].iteritems() for e in data]

        names = [n for (n, d) in samples['training'].iteritems()]

        out_tr, inp_tr = sample.unzip(inputs_tr)
        out_ts, inp_ts = sample.unzip(inputs_ts)

        out_tr = [singleton(names.index(n), len(names)) for n in out_tr]
        out_ts = [singleton(names.index(n), len(names)) for n in out_ts]
        return inp_tr, out_tr, inp_ts, out_ts

