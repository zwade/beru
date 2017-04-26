import wave
import math
import glob
import numpy as np
from scipy import fftpack
from array import array

BASE_CUTOFF = 3000

def sliding_window(array, size, fn):
    return [fn(array[max(0, i-size):min(len(array)-1, i+size)]) for i in range(len(array))]

def sliding_average(array, size):
    base = sum(array[:size*2+1])/(size*2+1)
    out = [array[i] for i in range(len(array))]
    for i in range(size, len(array)-size):
        out[i] = base
        base -= array[i-size]/(size*2+1)
        base += array[i+size]/(size*2+1)
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
            return fqs, dft

        data = np.abs(self.data)
        dft = np.fft.fft(data)[:len(data)/2]
        dft = np.abs(dft)
        dft = np.vectorize(math.log)(dft)
        dft = sliding_window(dft, 5, max)
        fqs = fftpack.fftfreq(len(data),float(1)/self.RATE)[:len(data)/2]
        
        self.fqs = fqs
        self.dft = dft
        return fqs, dft


    def get_smooth_data(self):
        fqs, dft = self.fetch_frequency_data()
        dft = sliding_average(dft, 70)
        return (fqs, dft)


    def get_data(self, points = 1024):
        fqs, dft = self.get_frequency_data()
        cutoff = np.searchsorted(fqs, BASE_CUTOFF)
        highest = max(dft[cutoff:])
        
        if type(points) == int:
            INTVL = min(len(dft)-cutoff, 20000)/points
            points = [cutoff + INTVL * i for i in range(points)]

        idxs = np.searchsorted(fqs, points)
        results = [(inline_avg(dft, i)/highest, j) 
                if i < len(dft) else None for (i,j) in zip(idxs, points)]
        results = filter(lambda x: x is not None, results)

        amps, frqs = unzip(results)

        return frqs, amps

    def time_divide_samples(self, points = 1024, size = 0.5):
        size = float(size)

        sample_len  = self.RATE*size
        num_samples = int(math.ceil(self.FRAMES/sample_len))
        data = [None for i in range(num_samples)]

        for i in range(num_samples):
            data[i] = Sample.from_data(self.data[int(i*sample_len):min(self.FRAMES, int((i+1)*sample_len))], self.RATE)

        return data


def get_all_samples(points = 1024):
    samples = {}
    for path in glob.glob("data/*/*.wav"):
        elements = path.split("/")
        sample_name = elements[1]
        sample = Sample.from_file(path)
        frqs, amps = sample.get_data(points)
        if sample_name not in samples:
            samples[sample_name] = (frqs, [])
        samples[sample_name][1].append(amps)
    return samples

if __name__ == "__main__":
    sample = Sample.from_file("data/x-pad/0.wav")
    print sample.get_data(10)
