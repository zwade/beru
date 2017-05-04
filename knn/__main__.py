from __future__ import print_function
import sample

import numpy as np
from six import iteritems
from six.moves import reduce


def classify(training, test, n = 5):
	test = np.array(test)
	vals = []
	for label in training:
		for l in training[label]:
			test_val = np.linalg.norm(np.array(l)-test)
			vals.append((test_val, label))

	best = {}
	for (v, l) in sorted(vals)[:n]:
		best[l] = 1 if l not in best else best[l] + 1

	(label, rank) = reduce(lambda x,y: x if (x[1] > y[1]) else y, iteritems(best)) 
	return label

def prep_data():
	data = sample.get_all_samples(96, version = sample.AUTOCORRELATION,  num_windows = 1, fraction = 1)
	training = data['training']
	testing  = data['test']

	total = 0
	num   = 0
	for label in testing:
		count = 0
		for test in testing[label]:
			best = classify(training, test, 8)
			if best == label:
				count += 1
		print ("{}: {}/{}".format(label, count, len(testing[label])))
		total += count
		num   += len(testing[label])
	print ("{}: {}/{}".format("Total", total, num))
	

if __name__ == '__main__':
	prep_data()
