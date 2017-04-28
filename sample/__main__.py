#!/usr/bin/env python2.7

from __future__ import print_function
import signal
import pyaudio
import wave
import glob
import sys
import os
import time
from struct import pack
from array import array

FORMAT = pyaudio.paInt16
RATE   = 44100
CHUNK  = 4096

dir_path = os.path.dirname(os.path.realpath(__file__))

colors = {"RED": "196", "BLUE": "21", "GREEN": "40", "ORANGE": "202"}
def color(string, color):
	if color in colors:
		c = colors[color]
	else:
		c = 16
	return "\033[38;5;{}m{}\033[0m".format(c, string)

def safe_int(string, default):
	try:
		return int(string)
	except ValueError:
		return default

def die(a,b):
	sys.exit(0)

signal.signal(signal.SIGINT, die) 

sys.stdout.write("Test Data? (n): ")
if raw_input() == "y":
	dir_path += "/test"

sample_files = glob.glob(dir_path+"/data/*/*.wav")
samples = {}
for name in sample_files:
	params = name.split("/")
	pattern = params[1]
	num = int(params[2].split(".")[0])
	if pattern not in samples:
		samples[pattern] = 0
	samples[pattern] = max(samples[pattern], num)

audio_inst = pyaudio.PyAudio()
stream = audio_inst.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

while True:
	sys.stdout.write("Sample name: ")
	name = raw_input()
	existing = 0
	if name in samples:
		print("{} has {} samples recorded. Adding more.".format(name, samples[name]+1))
		existing = samples[name]+1
	else:
		try:
			os.mkdir(dir_path+"/data/{}".format(name))
		except OSError:
			pass
	sys.stdout.write("Number to record (15): ")
	num = safe_int(raw_input(), 15)
	sys.stdout.write("Length of recording (3): ")
	length = safe_int(raw_input(), 3)

	print("Ready?" )
	time.sleep(1)
	sys.stdout.write(color("3...", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	sys.stdout.write(color("2...", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	sys.stdout.write(color("1...", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	print()
	for i in range(existing, existing+num):
		print("Draw a(n) {}".format(color(name, "BLUE")))
		
		r = array('h')
		t = 0
		while t < int(length * RATE):
			snd_data = array('h', stream.read(CHUNK, exception_on_overflow=False))
			r.extend(snd_data)
			t += len(snd_data)

		data = pack('<' + ('h'*len(r)), *r)
		wav_file = wave.open(dir_path+"/data/{}/{}.wav".format(name, i), "wb")
		wav_file.setnchannels(1)
		wav_file.setsampwidth(audio_inst.get_sample_size(FORMAT))
		wav_file.setframerate(RATE)
		wav_file.writeframes(data)
		wav_file.close()
		print(color("Good!", "GREEN"))
		time.sleep(0.5)


