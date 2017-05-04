import wave
import math
import glob
import numpy as np
import os
from scipy import fftpack
from array import array
from multiprocessing import Pool

MIN_FREQ = 20000
MAX_FREQ = 40000

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


FREQUENCY = 1
AUTOCORRELATION = 2
class Sample:
	def __init__(self):
		self.data = None
		self.RATE = None
		self.fqs  = None
		self.dft  = None
		self.xcor = None

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

	def get_autocorrelation_data(self):
		if self.xcor is not None:
			return self.xcor

		xcor = (np.fft.ifft(abs(np.array(np.fft.fft(self.data)))**2))
		xcor = abs(xcor[:len(xcor)//2])
		self.xcor = xcor
		return xcor

	def get_frequency_data(self):
		if self.fqs is not None and self.dft is not None:
			return self.fqs, self.dft

		data = np.abs(self.data)
		dft = np.fft.fft(data)[:len(data)//2]
		dft = np.abs(dft)
		dft = np.vectorize(lambda x: math.log(x + 1))(dft)
		fqs = fftpack.fftfreq(len(data),float(1)/self.RATE)[:len(data)//2]

		self.fqs = fqs
		self.dft = dft
		return fqs, dft


	def get_smooth_data(self):
		fqs, dft = self.get_frequency_data()
		dft = sliding_average(dft, 70)
		return (fqs, dft)


	def get_data(self, buckets = 1024, version = FREQUENCY):
		data = []
		if version == AUTOCORRELATION:
			xcor  = self.get_autocorrelation_data()
			data  = np.array(xcor[4:buckets+4])

		elif version == FREQUENCY:
			fqs, dft = self.get_frequency_data()
			start = np.searchsorted(fqs, MIN_FREQ)
			end = np.searchsorted(fqs, MAX_FREQ)
			data = np.array(dft[start:end])
			data = np.array_split(data, buckets)
			data = [np.sum(np.multiply(np.hamming(b.size), b)) for b in data]

		return data

	def time_divide_samples(self, points = 1024, num_windows = 10):
		parts = np.array_split(self.data, num_windows)
		data = [Sample.from_data(part, self.RATE) for part in parts]

		return data
	
	def get_subtime_data(self, points, num_windows, version):
		current = np.array([])

		for s in self.time_divide_samples(points, num_windows):
			amps = s.get_data(points, version)
			current = np.concatenate([current, amps])

		average = np.average(current)
		current = current - average
		scale = np.max(np.absolute(current))
		current = current / scale

		return current


def load(path):
	global GL_POINTS, GL_NUM_WINDOWS, GL_VERSION
	points = GL_POINTS
	num_windows = GL_NUM_WINDOWS
	version = GL_VERSION
	print("Loading", path)
	elements = path.split("/")
	sample_name = elements[-2]
	sample = Sample.from_file(path)

	return (sample_name, sample.get_subtime_data(points, num_windows, version))

def get_all_in_path(p, points = 1024, num_windows = 10, fraction = 1, version = FREQUENCY):
	global GL_POINTS, GL_NUM_WINDOWS, GL_VERSION
	GL_POINTS = points
	GL_NUM_WINDOWS = num_windows
	GL_VERSION = version
	samples = {}
	paths = sorted(glob.glob(p))[::fraction]

	with Pool(8) as p:
		loaded = p.map(load, paths)

	for (name, data) in loaded:
		if name not in samples:
			samples[name] = []
		samples[name].append(data)

	return samples

def get_all_samples(points = 1024, num_windows = 10, fraction = 1, version = FREQUENCY):
	return {
		'training': get_all_in_path(dir_path + "/data/*/*.wav",  points, num_windows, fraction, version),
		'test': get_all_in_path(dir_path + "/test/data/*/*.wav", points, num_windows, 1, version) 
	}

if __name__ == "__main__":
	sample = Sample.from_file(dir_path + "/data/x-pad/0.wav")
