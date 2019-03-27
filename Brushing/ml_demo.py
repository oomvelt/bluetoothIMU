import numpy
from scipy import signal
import matplotlib.pyplot as plt

np = numpy.loadtxt("HMP_Dataset/Brush_teeth/Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt")

print(np)


#real_val = -1.5g + (coded_val/63)*3g

real = (np/63)*3 -1.5

print(real[:, 0])

print(len(real[:, 0]))

#hyperoptimization parameters: stft window size, nperseg,
#and event frequency and amplitude thresholds

f, t, Sxx = signal.spectrogram(real[:, 0], fs=32, nperseg=16)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

# Data for plotting
t = numpy.arange(len(real[:, 0]))#/32 #sample rate is 32 hz
s = real[:, 0]

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='accel (g)',
       title='Brushin Ma Teefs')
ax.grid()

#fig.savefig("test.png")
plt.show()


#TODO: pick a STFT window that optimally smooths the data. - nperseg=16 works pretty well, at least on
#the first test case.
#TODO: Collect events, and try highlighting event windows back on the time series chart.
#Apply FFT to each window, do peak selection as feature extraction. Train NN on number of peaks,4
#peak frequency, relative peak amplitude, maybe spacing between peaks?


