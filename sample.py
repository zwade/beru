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

def get_raw_data(path):
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

    data = np.abs(data)
    dft = np.fft.fft(data)[:len(data)/2]
    dft = np.abs(dft)
    dft = np.vectorize(math.log)(dft)
    dft = sliding_window(dft, 5, max)
    #dft = sliding_window(dft, 70, lambda x: sum(x)/len(x))
    dft = sliding_average(dft, 70)
    fqs = fftpack.fftfreq(len(data),float(1)/RATE)[:len(data)/2]
    return (fqs, dft)


def get_data(path, points = 1024):
    fqs, dft = get_raw_data(path)
    highest = max(dft[BASE_CUTOFF:])
    
    if type(points) == int:
        INTVL = min(len(dft)-BASE_CUTOFF, 20000)/points
        points = [BASE_CUTOFF + INTVL * i for i in range(points)]

    idxs = np.searchsorted(fqs, points)
    results = [(dft[i]/highest, j) if i < len(dft) else None for (i,j) in zip(idxs, points)]
    results = filter(lambda x: x is not None, results)

    amps, frqs = unzip(results)

    return frqs, amps

def get_all_samples(points = 1024):
    samples = {}
    for path in glob.glob("data/*/*.wav"):
        elements = path.split("/")
        sample_name = elements[1]
        frqs, amps = get_data(path, points)
        if sample_name not in samples:
            samples[sample_name] = (frqs, [])
        samples[sample_name][1].append(amps)
    return samples

if __name__ == "__main__":
    print get_data("data/x-pad/0.wav", 10)
