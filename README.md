# Berü

Berü is a set of tools for training and detecting gestures made on the chassis of a laptop. It accomplishes this by using a Time Delay Neural Network (TDNN) trained on a large set of data.

## Installation

This project used [Anaconda3](https://www.continuum.io/downloads) to manage dependencies. It is recommended that you use it. If you choose not to, you will need to manually install `numpy` and `scipy`.

You will also need the `pyaudio` package. Before you can install it however, you will need `portaudio` installed. To do this, run

#### OSX

```bash
brew install portaudio
```

#### Ubuntu

```bash
sudo apt-get install portaudio19-dev
```

Then, run `pip install pyaudio`. As of writing, Anaconda does not have the latest version of `pyaudio`, so it is recommended that you install via `pip`.

## Sample Acquisition

To acquire more testing or training samples, run `python sample`. You will then be prompted to add testing or training data, the name of samples, the number to record, and their duration. 

## Neural Net Configuration

To configure the Neural Network, edit `neural-net/load_data.py`. The parameters that you will need to change are 

 - `NUM_FQS` => The total number of samples
 - `NUM_TIME` => The number of time buckets
 - `VERSION` => Either `sample.FREQUENCY` or `sample.AUTOCORRELATION` depending on which method you want to use
 
## Neural Net Creation

To create a new neural net to train on, run `python neural-net init <name> <layer1> [...<layerN>]` where `name` can be anything, each layer should be of the form `<NODES>,<TIME_OFFSETS>`. Furthermore, `layer1` should always have `NUM_FREQ` nodes, and `NUM_TIME` time offsets, and `layern` should have nodes equal to the number of samples, and one time offset. 

## Neural Net Training

To train the newly created neural network, run `python neural-net init <input_name> <output_name>` where `<input_name>` is the name specified above, and `<output_name>` is the file to store the resulting weights. You can also specifiy the `-t` flag to also test the net, `-i N` to perform testing every `N` rounds, `-l` to continue training instead of stopping after `N` rounds, and `-r RATE` to set the learning rate. This will display a graphical interface that shows the progress that's being made, with a breakdown of wherein error lies. You can hit `ctrl-c` to save and exit at any time.

## Nerual Net Classification

Finally, to test the trained data against a live input stream, run `python neural-net classify <weight_file> <FRQ|COR>` where `weight_file` is the output file from training, and they type is `FRQ` if the net was trained with `sample.FREQUENCY` and `COR` if it was trained with `sample.AUTOCORRELATION`. 

## Credits

This was designed and implemented by Matthew Savage (@bluepichu) and Zachary Wade (@zwade). 
