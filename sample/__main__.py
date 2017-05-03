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
from builtins import input
from random import randint

FORMAT = pyaudio.paInt16
RATE   = 96000
CHUNK  = 4096
LENGTH = 2
GESTURE_LENGTH = 1

dir_path = os.path.dirname(os.path.realpath(__file__))

colors = {"RED": "31", "BLUE": "34", "GREEN": "32", "ORANGE": "33"}
def color(string, color):
	if color in colors:
		c = colors[color]
	else:
		c = 16
	return "\033[{}m{}\033[0m".format(c, string)

def safe_int(string, default):
	try:
		return int(string)
	except ValueError:
		return default

def die(a,b):
	sys.exit(0)

signal.signal(signal.SIGINT, die)

sys.stdout.write("Test Data? (n): ")
if input() == "y":
	dir_path += "/test"

sample_files = glob.glob(dir_path+"/data/*/*.wav")
samples = {}
for name in sample_files:
	params = name.split("/")
	pattern = params[-2]
	num = int(params[-1].split(".")[0])
	if pattern not in samples:
		samples[pattern] = 0
	samples[pattern] = max(samples[pattern], num)

audio_inst = pyaudio.PyAudio()
stream = audio_inst.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

while True:
	sys.stdout.write("Sample name: ")
	name = input()
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
	num = safe_int(input(), 15)

	print("Ready?" )
	time.sleep(1)
	sys.stdout.write(color("3... ", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	sys.stdout.write(color("2... ", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	sys.stdout.write(color("1... ", "RED"))
	sys.stdout.flush()
	time.sleep(1)
	print()
	for i in range(existing, existing+num):
		r = array('h')
		
		print(color("Recording noise", "ORANGE"))

		t = 0
		while t < int((LENGTH - GESTURE_LENGTH) * RATE):
			snd_data = array('h', stream.read(CHUNK, exception_on_overflow=False))
			r.extend(snd_data)
			t += len(snd_data)

		print("Draw a(n) {}".format(color(name, "BLUE")))

		t = 0
		while t < int(GESTURE_LENGTH * RATE):
			snd_data = array('h', stream.read(CHUNK, exception_on_overflow=False))
			r.extend(snd_data)
			t += len(snd_data)

		print(color("Recording noise", "ORANGE"))

		t = 0
		while t < int((LENGTH - GESTURE_LENGTH) * RATE):
			snd_data = array('h', stream.read(CHUNK, exception_on_overflow=False))
			r.extend(snd_data)
			t += len(snd_data)

		start = randint(0, int((LENGTH - GESTURE_LENGTH) * RATE))
		r = r[start : start + int(LENGTH * RATE)]
		data = pack('<' + ('h'*len(r)), *r)
		wav_file = wave.open(dir_path+"/data/{}/{}.wav".format(name, i), "wb")
		wav_file.setnchannels(1)
		wav_file.setsampwidth(audio_inst.get_sample_size(FORMAT))
		wav_file.setframerate(RATE)
		wav_file.writeframes(data)
		wav_file.close()
		print(color("Good!", "GREEN"))
		time.sleep(0.5)


