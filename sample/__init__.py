import wave
import math
import glob
import numpy as np
import os
from scipy import fftpack
from array import array

BASE_CUTOFF = 3000

dir_path = os.path.dirname(os.path.realpath(__file__))

def sliding_window(array, size, fn):
	return [fn(array[max(0, i-size):min(len(array)-1, i+size)]) for i in range(len(array))]

def sliding_average(array, size):
	base = sum(array[:size*2+1])//(size*2+1)
	out = [array[i] for i in range(len(array))]
	for i in range(size, len(array)-size):
		out[i] = base
		base -= array[i-size]//(size*2+1)
		base += array[i+size]//(size*2+1)
	return out

def unzip(a):
	return ([x for (x,y) in a], [y for (x,y) in a])

def inline_avg(array, idx, a_length = 70):
	lower = max(0, idx-a_length)
	upper = min(len(array), idx+a_length)

	return sum(array[lower:upper])/(upper-lower)


class Sample:
	def __init__(self):
		self.data = None
		self.RATE = None
		self.fqs  = None
		self.dft  = None

	@classmethod
	def from_file(this, path):
		self = this()
		wave_file = wave.open(path, "rb")
		WIDTH  = wave_file.getsampwidth()
		RATE   = wave_file.getframerate()
		FRAMES = wave_file.getnframes()

		CHUNK_SIZE = 1024

		data = array('h')
		read = 0

		while read < FRAMES:
			snd_data = array('h', wave_file.readframes(CHUNK_SIZE))
			read += len(snd_data)
			data.extend(snd_data)
		
		self.RATE   = RATE
		self.FRAMES = FRAMES
		self.data   = data

		return self

	@classmethod
	def from_data(this, data, RATE):
		self = this()
		self.data   = data
		self.RATE   = RATE
		self.FRAMES = len(data)

		return self

	def get_frequency_data(self):
		if self.fqs is not None and self.dft is not None:
			return self.fqs, self.dft

		data = np.abs(self.data)
		dft = np.fft.fft(data)[:len(data)//2]
		dft = np.abs(dft)
		dft = np.vectorize(lambda x: math.log(x + 1))(dft)
		dft = sliding_window(dft, 5, max)
		fqs = fftpack.fftfreq(len(data),float(1)/self.RATE)[:len(data)//2]
		
		self.fqs = fqs
		self.dft = dft
		return fqs, dft


	def get_smooth_data(self):
		fqs, dft = self.get_frequency_data()
		dft = sliding_average(dft, 70)
		return (fqs, dft)


	def get_data(self, buckets = 1024):
		fqs, dft = self.get_frequency_data()
		start = np.searchsorted(fqs, BASE_CUTOFF)
		end = np.searchsorted(fqs, 20000)
		data = np.array(dft[start:end])
		data = np.array_split(data, buckets)
		data = [np.sum(np.multiply(np.hamming(b.size), b)) for b in data]
		return data

	def time_divide_samples(self, points = 1024, size = 0.5):
		size = float(size)

		sample_len  = self.RATE*size
		num_samples = int(math.ceil(self.FRAMES/sample_len))
		data = [None for i in range(num_samples)]

		for i in range(num_samples):
			data[i] = Sample.from_data(self.data[int(i*sample_len):min(self.FRAMES, int((i+1)*sample_len))], self.RATE)

		return data

def get_all_in_path(p, points = 1024, bucket_len = 0.5, fraction = 1):
	samples = {}
	for path in sorted(glob.glob(p))[::fraction]:
		elements = path.split("/")
		sample_name = elements[-2]
		sample = Sample.from_file(path)
		current = np.array([])

		for s in sample.time_divide_samples(points, bucket_len):
			amps = s.get_data(points)
			current = np.concatenate([current, amps])

		average = np.average(current)
		current = current - average
		scale = np.max(np.absolute(current))
		current = current / scale

		if sample_name not in samples:
			samples[sample_name] = ([], [])
		samples[sample_name][1].append(current)
	return samples

def get_all_samples(points = 1024, bucket_len = 0.5, fraction = 1):
	return {
		'training': get_all_in_path(dir_path + "/data/*/*.wav",  points, bucket_len, fraction),
		'test': get_all_in_path(dir_path + "/test/data/*/*.wav", points, bucket_len)
	}

if __name__ == "__main__":
	sample = Sample.from_file(dir_path + "/data/x-pad/0.wav")
