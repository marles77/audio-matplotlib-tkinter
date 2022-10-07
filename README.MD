# AudioPlot
Use Python 3, Matplotlib, Sounddevice, and Tkinter to visualize and play audio files while tracking the current audio position.

## Aim
To create a simple app which will allow to 1) open an audio file, 2) display the sound data, 3) play it while displaying the current position on the track, and 4) put some markers. So, a bit like [Audacity](https://www.audacityteam.org/) or similar apps.

## Problem
1. In order to dynamically visualize audio data and show a tracking marker we need to update the image at a reasonable rate. Unfortunately, Matplotlib library for Python is rather slow, and if you want to update the plot frequently, say at 50 frames/second, it may be impossible to achieve. Which is why the implementation of [this solution based on sequential plotting](https://blog.devgenius.io/easy-audio-visualization-python-d3c6aa935212) will likely end up with a failure. There are several alternative approaches described e.g. [here](https://bastibe.de/2013-05-30-speeding-up-matplotlib.html). 
2. A couple of Python libraries have been created to deal with audio data e.g. [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/), [PyDub](http://pydub.com/), [PyGame](https://www.pygame.org/docs/ref/mixer.html). However, their usage can be quite challanging when it comes to some details (e.g. difficult PyDub setup at least on Windows, impossible to set sound position for WAV files in PyGame).  

## Solution
1. If you dig deeper into Matplotlib, it turns out to be quite flexible, but you need to work on different levels. It is crucial to understand that rendering graphics in Matplotlib involves different layers and its API lets you access these layers to improve graphics performance; [more about it in this tutorial](https://matplotlib.org/stable/tutorials/intermediate/artists.html). So, usually, it is enough when we just operate on higher level objects like containers (figure, axes) and artists (lines, rectangles, ticks, etc.). However, when high performance is needed, e.g. when frequent update of figure content is necessary, it is advised to play with the low level canvas object, on which everything else lays. The [above-mentioned](https://bastibe.de/2013-05-30-speeding-up-matplotlib.html) high-performance solutions require accessing low-level objects. Note, however, that they may work differently depending on a chosen backend. Some solutions that work with the Qt backend will not work if you want to use e.g. Tkinter. It seems that the best and the most versatile working solution is the one using [blitting](https://matplotlib.org/stable/tutorials/advanced/blitting.html) which generally assumes that there are parts of the canvas that are constant and can just be copied, stored in a memory buffer as a "background", and then restored when needed (e.g. in a subsequent frame). So, we only draw artists that are changing their properties (e.g. location, shape, color). In this way from up to 10 frames/second (depending on the number of data points in a plot) we can reach 50 and more frames/second.
2. When it comes to audio file replay, the best performance and best access to the raw data can be obtained if the sound-dealing library is based on NumPy. [Python-Sounddevice](https://python-sounddevice.readthedocs.io/en/0.4.5/) is a simple library which gives you a nice feeling of being in full control over what is going on when working with sound files. Given NumPy's high performance due to its good use of system resources, it seems a reasonable choice from the perspective of the current purpose. 

## Dependencies
Install dependencies with:
`pip install --upgrade numpy matplotlib soundfile sounddevice`

## Usage
Run the script: `python main.py`

## Features
The script (written in Python 3.9) creates a simple GUI that allows to open an audio file (WAV), view the audio track, and play the file while tracking the current playback progress on a plot.

## Limitations and known issues
The script works fine on Windows 10.