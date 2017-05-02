from __future__ import print_function
import sample

import numpy as np
from six import iteritems
from six.moves import reduce


def classify(training, test, n = 5):
	test = np.array(test)
	vals = {}
	for label in training:
		vals[label] = 0
		for val in training[label][1]:
			test_val = np.linalg.norm(np.array(val)-test)
			vals[label] += test_val
			print("  - [{}] Diff: {:1.4f}".format(label, test_val))
		vals[label] /= len(training[label])

	bestL = ""
	bestV = 0 
	(label, rank) = reduce(lambda x,y: x if (x[1] < y[1]) else y, iteritems(vals)) 
	return label

def prep_data():
	data = sample.get_all_samples(128, )
	training = data['training']
	testing  = data['test']
	#training = [("first", [.88,.1,.05]), ("first", [.90, .4, .3]), ("second", [.2,.7,.05]), ("second", [0.05,.95,0.1]), ("third", [0,.1, .8])]

	#testing = [("first", [0.76, 0.3, 0.1]), ("second", [0,1,0]), ("third", [0,0,0.8])]

	for label in testing:
		for test in testing[label][1]:
			print ("{}:".format(label))
			best = classify(training, test)
			print ("    {} {}".format(label, best))
			break
	

if __name__ == '__main__':
	prep_data()
