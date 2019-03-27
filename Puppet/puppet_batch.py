import time
import serial
import numpy
import pandas as pd
from scipy import signal
from scipy import fftpack
import math
import matplotlib
import matplotlib.pyplot as plt


def sync_to_uq(ser):
    """
    Aligns the serial port read to the next 'UQ' header at the start of a data frame.

    :type ser: pySerial Serial object
    :param ser: The opened pySerial port bound to the Bluetooth connection with the BWT901CL.
    :rtype: Bool, bytestring
    :return: return the synced state (should be True on return), and the 9 bytes of 'UQ' accel data
    """

    synced = False
    u_flag = False

    # Inner loop discards bytes until we align with the next 'UQ' header.
    while not synced:

        byte = ser.read(1)

        # We haven't set the u_flag yet and we just hit a 'U' byte
        if not u_flag and byte == b'U':
            u_flag = True
        # The last byte was 'U' and this byte is 'Q', so we're done
        elif u_flag and byte == b'Q':
            synced = True
        # Either u_flag wasn't set and this byte isn't 'U', or
        # u_flag was set and this byte wasn't 'Q'
        # either way, reset the state flags.
        else:
            u_flag = False
            synced = False

    # Since we've aligned to the 'UQ' header, grab and return the 'UQ' accels.
    # If we desynced on the previous measurement, no sense in throwing away a
    # second data point while we're aligning.
    uq_bytes = ser.read(9)

    return synced, uq_bytes


def acquire_data(accel, w_vel, mag_angle, sample_freq, capture_timeout):
    """
    Acquires accelerometer data and stores it in arrays.

    :type accel: 2D array of float
    :param accel: array of accelerometer measurements
    :type w_vel: 2D array of float
    :param w_vel: array of gyro angular velocities.
    :type mag_angle: 2D array of float
    :param mag_angle: array of magnetometer angles.
    :type sample_freq: int
    :param sample_freq: Sampling frequency in Hz
    :type capture_timeout: int
    :param capture_timeout: the amount of time to capture data for, in seconds
    :return: None
    """

    print("Starting data capture...")
    start = time.time()
    current = start
    synced = False
    desync_count = 0

    with serial.Serial('/dev/rfcomm0', 115200, timeout=100) as ser:

        # Flushing the serial input buffer because why not
        ser.reset_input_buffer()

        # Once we've opened the serial port and flushed the buffer, do a first alignment and
        # initialize last_realtime froM the hardware clock.

        synced, a_raw = sync_to_uq(ser)
        last_realtime = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)

        # Outer loop captures until timeout
        while (current - start) < capture_timeout:

            current = time.time()

            # If this is the first iteration or we desynced on the last iteration, realign to the next 'UQ' header.
            if not synced:
                # Wait on align to sync. TODO: Alignment should have a timeout for squawk-and-stop.
                synced, a_raw = sync_to_uq(ser)

                a_1_int = int.from_bytes(a_raw[0:2], byteorder='little', signed=True)
                a_2_int = int.from_bytes(a_raw[2:4], byteorder='little', signed=True)
                a_3_int = int.from_bytes(a_raw[4:6], byteorder='little', signed=True)
            # Otherwise just attempt to read the 'UQ' frame and get the accel data.
            else:
                s = ser.read(11)
                if s[0:2] != b'UQ':
                    # Commenting out these desyncs, because we already know we are getting desynced in
                    # nominal operation. NaN counting statistics in detect_events() give roughly equivalent
                    # diagnostic information, and don't cause the real time performance hit of writing
                    # to stdout.

                    # print("Desynced! " + str(s[0:2]))
                    synced = False
                    continue
                else:
                    a_1_int = int.from_bytes(s[2:4], byteorder='little', signed=True)
                    a_2_int = int.from_bytes(s[4:6], byteorder='little', signed=True)
                    a_3_int = int.from_bytes(s[6:8], byteorder='little', signed=True)

            # Read the 'UR' frame and get the gyro angular velocity data.
            s = ser.read(11)
            if s[0:2] != b'UR':
                # print("Desynced! " + str(s[0:2]))
                synced = False
                desync_count += 1
                continue
            else:
                w_1_int = int.from_bytes(s[2:4], byteorder='little', signed=True)
                w_2_int = int.from_bytes(s[4:6], byteorder='little', signed=True)
                w_3_int = int.from_bytes(s[6:8], byteorder='little', signed=True)

            # Read the 'US' frame and get the magnetometer angles.
            s = ser.read(11)
            if s[0:2] != b'US':
                # print("Desynced! " + str(s[0:2]))
                synced = False
                desync_count += 1
                continue
            else:
                angle_1_int = int.from_bytes(s[2:4], byteorder='little', signed=True)
                angle_2_int = int.from_bytes(s[4:6], byteorder='little', signed=True)
                angle_3_int = int.from_bytes(s[6:8], byteorder='little', signed=True)

            # I haven't been able to figure out what the data is in the 'UT' frame
            s = ser.read(11)
            if __name__ == '__main__':
                if s[0:2] != b'UT':
                    # print("Desynced! " + str(s[0:2]))
                    synced = False
                    desync_count += 1
                    continue

            # We made it to the end of one complete frame, so we are assuming that this is a good data point.
            # Before capturing this data, we need to check if a desync has caused data dropout. If so, we'll
            # fill with NaN.

            realtime = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)

            # If the elapsed time since we last captured a good data point is greater than the BWT901CL's
            # sampling period, then we lost one or more data points to a desync.

            time_delta = int(round((realtime - last_realtime) / (1.0 / sample_freq)))

            # ...and update last_realtime to the current realtime, for use on the next cycle.

            last_realtime = realtime

            # If we lost data, append NaN's for the missing data points, then append the new good data point.
            # If we didn't lose data, just append the new good data point.

            if time_delta > 1:

                #print("DEBUG: time_delta: " + str(time_delta))

                for i in range(time_delta - 1):

                    for idx in range(3):

                        accel[idx].append(math.nan)
                        w_vel[idx].append(math.nan)
                        mag_angle[idx].append(math.nan)

            for idx, value in enumerate([a_1_int, a_2_int, a_3_int]):
                accel[idx].append(value / 32768.0 * 16)

            for idx, value in enumerate([w_1_int, w_2_int, w_3_int]):
                w_vel[idx].append(value / 32768.0 * 2000)

            for idx, value in enumerate([angle_1_int, angle_2_int, angle_3_int]):
                mag_angle[idx].append(value / 32768.0 * 180)

    print("Data capture completed.")
    print("Desync count: " + str(desync_count))
    print("Data points collected: " + str(len(accel[0])))


def detect_events(accel):
    """
    Does FFT based event detection and plotting.

    :type accel: 1D array of float
    :param accel: An array of acceleration measurements from the accelerometer, along a single axis
    :return: None
    """

    df = pd.DataFrame(data=accel[0], columns=['x'])

    print("Number of NaN points: " + str(df.isna().sum()))

    # Doing a linear interpolation that replaces the NaN placeholders for desynced, dropped data
    # with a straight line between the last good data point before a dropout and the next good data
    # point. This does seem to improve the signal-to-noise ratio in the FFT.
    df = df.interpolate()

    # hyper-optimization parameters: stft window size, nperseg,
    # and event frequency and amplitude thresholds

    # stft; doesn't seem to work as well as a spectrogram.
    # f, t, Zxx = signal.stft(df.loc[:, 'x'], fs=32, nperseg=16)

    f, t, sxx = signal.spectrogram(df.loc[:, 'x'], fs=10, nperseg=16, mode='magnitude')

    # plot for spectrogram output
    fig, ax = plt.subplots()

    ax.set(xlabel='time spectrogram', ylabel='frequency bins',
           title='Brushing Spectrogram')
    plt.pcolormesh(t, f, sxx)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')

    hz_bin_4 = pd.DataFrame(sxx[2, :].reshape(len(sxx[2, :]), 1), columns=['4hz amplitude'])

    # Uncomment this to check the raw amplitudes of the 4 hz bin, for determining the threshold
    # value below.
    print("DEBUG: 4hz amplitude: ")
    print(hz_bin_4['4hz amplitude'])

    hz_bin_4['4 hz threshold?'] = numpy.where(hz_bin_4['4hz amplitude'] > 2, 1, 0)

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

    # Data for plotting
    t = numpy.arange(len(df.loc[:, 'x']))  # /10 if sample rate is 10 hz
    s = df.loc[:, 'x']

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (10 hz counts)', ylabel='accel (g)',
           title='Brushing Accelerometer Data')
    ax.grid()

    win_start = event_windows[1][0]
    win_end = event_windows[1][1]

    assert(win_end > win_start)

    print(win_end, win_start)

    yf = fftpack.rfft(df.loc[win_start:win_end, 'x'])

    # Is this a better way of getting the power spectrum than numpy.abs(yf?)
    #yf = (yf * yf.conj()).real
    yf = numpy.abs(yf)

    xf = fftpack.rfftfreq((win_end - win_start + 1), 1./10)

    fig, ax = plt.subplots()
    # plot for spectrogram output
    ax.set(xlabel='Frequency (Hz)', ylabel='Power Spectrum Amplitude',
           title='Peak Detection on Event')

    plt.plot(xf, yf)

    peaks, peaks_dict = signal.find_peaks(yf, height=100)

    print(peaks, peaks_dict)
    plt.plot(xf[peaks], yf[peaks], "x")

    # Note: would need some way to normalize amplitude of the power spectrum to use peak amplitude
    # as a feature. Right now, it looks like the amplitude is a function of the duration of the
    # periodic signal.

    # Data for plotting
    t = numpy.arange(len(df.loc[:, 'x']))  # /10 if sample rate is 10 hz
    s = df.loc[:, 'x']

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (10 hz counts)', ylabel='accel (g)',
           title='Event Detection on Brushing Accelerometer Data')
    ax.grid()

    for item in event_windows:
        plt.axvspan(item[0], item[1], color='yellow', alpha=0.5)

    plt.show()


def main():
    """
    Main function acquires data from the BWT901CL for a fixed length of time, and then performs a FFT analysis on it.

    :return: None
    """

    accel = [[] for i in range(3)]
    w_vel = [[] for i in range(3)]
    mag_angle = [[] for i in range(3)]
    default_sample_freq = 50
    capture_timeout = 30

    acquire_data(accel, w_vel, mag_angle, default_sample_freq, capture_timeout)

    #TODO: update detect_events to take default_sample_freq as an argument,
    #instead of hardcoding 10 hz in there.
    detect_events(w_vel)

if __name__ == "__main__":
    main()
