import sample

import numpy as np
from six import iteritems


def classify(training, test, n = 5):
	dt = [("label", "S20"), ("count", np.float64)]
	out = np.array([("string", 0) for i in range(len(training))], dtype=dt)
	idx = 0
	test = np.array(test)
	for (label, datum) in training:
		test_val = np.linalg.norm(np.array(datum)-test)
		out[idx] = (label, test_val)
		idx += 1

	n_elts = np.sort(out, order='count')[:n]
	labels, data = sample.unzip(n_elts)
	pair = zip(*np.unique(labels, return_counts = True))
	pair = np.array(pair, dtype=dt)

	label, datum = np.sort(pair, order='count')[-1]
	return label

def form_data(data):
	return [(label, datum) for (label, (fqs, data_elts)) in iteritems(data) for datum in data_elts]
	
def prep_data():
	data = sample.get_all_samples(128, )
	training = form_data(data['training'])
	testing  = form_data(data['test'])
	#training = [("first", [.88,.1,.05]), ("first", [.90, .4, .3]), ("second", [.2,.7,.05]), ("second", [0.05,.95,0.1]), ("third", [0,.1, .8])]

	#testing = [("first", [0.76, 0.3, 0.1]), ("second", [0,1,0]), ("third", [0,0,0.8])]

	for (label, test) in testing:
		best = classify(training, test, 7)
		print (label, best)
	

if __name__ == '__main__':
	prep_data()
