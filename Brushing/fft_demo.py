import numpy
import pandas as pd
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import math

# This demo requires the HMP Dataset, which can be downloaded here:

# https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

# HMP_Dataset/Brush_teeth/Accelerometer-2011-04-11-13-29-54-brush_teeth-f1.txt
# HMP_Dataset/Brush_teeth/Accelerometer-2011-05-30-10-34-16-brush_teeth-m1.txt

df = pd.read_csv("HMP_Dataset/Brush_teeth/Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt", delim_whitespace=True,
                 header=None, names=['x', 'y', 'z'])

# df = pd.read_csv("HMP_Dataset/Brush_teeth/Accelerometer-2011-05-30-10-34-16-brush_teeth-m1.txt",
#                   delim_whitespace=True, header=None, names=['x', 'y', 'z'])

df = (df/63)*3 - 1.5

# hyperoptimization parameters: stft window size, nperseg,
# and event frequency and amplitude thresholds

# stft; doesn't seem to work as well as a spectrogram.
# f, t, Zxx = signal.stft(df.loc[:, 'x'], fs=32, nperseg=16)

f, t, Sxx = signal.spectrogram(df.loc[:, 'x'], fs=32, nperseg=16, mode='magnitude')

fig, ax = plt.subplots()
# plot for spectrogram output
ax.set(xlabel='time spectrogram', ylabel='frequency bins',
       title='Brushing Spectrogram')
plt.pcolormesh(t, f, Sxx)
# plt.pcolormesh(t, f, numpy.abs(Zxx), vmin=0, vmax=df['x'].max())
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')

hz_bin_4 = pd.DataFrame(Sxx[2, :].reshape(len(Sxx[2, :]), 1), columns=['4hz amplitude'])
hz_bin_4['4 hz threshold?'] = numpy.where(hz_bin_4['4hz amplitude'] > .2, 1, 0)

# Split the DataFrame into consecutive sequences of all 0 (below threshold) and all 1 ( above threshold)
# https://stackoverflow.com/questions/40802800/pandas-dataframe-how-to-groupby-consecutive-values
raw_groups = hz_bin_4.groupby([(hz_bin_4['4 hz threshold?'] != hz_bin_4['4 hz threshold?'] .shift()).cumsum()])

# Check that all threshold? elements in each split Dataframe are either all 1 or all 0.
for group in raw_groups:
    assert(group[1]['4 hz threshold?'].all() or not group[1]['4 hz threshold?'].any())

# Build a list of the split DataFrames
events = [item for _, item in raw_groups if (item['4 hz threshold?'].all() and len(item['4 hz threshold?']) >= 3)]

# The spectrogram has a different number of time data points since it applies sliding time windows over the
# original data, so we need to figure out a scaling to go to the original number of accelerometer measurement points
time_scale = len(df.loc[:, 'x'])/len(hz_bin_4['4 hz threshold?'])

print("time_scale: " + str(time_scale))
event_windows = [(math.floor(item.index.min() * time_scale), math.ceil(item.index.max() * time_scale))
                 for item in events]

print(event_windows)

win_start = event_windows[0][0]
win_end = event_windows[0][1]

assert(win_end > win_start)

print(win_end, win_start)

yf = fftpack.rfft(df.loc[win_start:win_end, 'x'])

# Is this a better way of getting the power spectrum than numpy.abs(yf?)
yf = (yf * yf.conj()).real

xf = fftpack.rfftfreq((win_end - win_start + 1), 1./32)

fig, ax = plt.subplots()
#plot for spectrogram output
ax.set(xlabel='Frequency (Hz)', ylabel='Power Spectrum Amplitude',
       title='Peak Detection on Event')

plt.plot(xf, yf)


peaks, peaks_dict = signal.find_peaks(yf, height=50)

print(peaks, peaks_dict)
plt.plot(xf[peaks], yf[peaks], "x")

# Note: would need some way to normalize amplitude of the power spectrum to use peak amplitude
# as a feature. Right now, it looks like the amplitude is a function of the duration of the
# periodic signal.

# Data for plotting
t = numpy.arange(len(df.loc[:, 'x']))#/32 #sample rate is 32 hz
s = df.loc[:, 'x']

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (32 hz counts)', ylabel='accel (g)',
       title='Brushing Accelerometer Data')
ax.grid()

for item in event_windows:
    plt.axvspan(item[0], item[1], color='yellow', alpha=0.5)

plt.show()




