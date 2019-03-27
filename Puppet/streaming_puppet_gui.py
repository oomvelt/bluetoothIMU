from multiprocessing import Process, Queue
import time
import serial
import numpy
import pandas as pd
from scipy import signal
from scipy import fftpack
import math
import sys
import PySimpleGUI as sg


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


def acquire_data(sample_freq, capture_window, queue):
    """
    Acquires accelerometer data and stores it in arrays.

    :type sample_freq: int
    :param sample_freq: Sampling frequency in Hz
    :type capture_window: int
    :param capture_window: the number of data points to capture before writing to the queue. If acquire_data()
                           happens to desync at the end of the capture window, it may return more data points
                           than requested.
    :type queue: Multiprocessing Queue
    :param queue: A multiprocessing queue that allows a dict containing a captured batch of measurements to be
                  passed to detect_events().
    :return: None
    """

    print("Starting data capture...")

    accel = [[] for i in range(3)]
    w_vel = [[] for i in range(3)]
    mag_angle = [[] for i in range(3)]
    desync_count = 0
    capture_count = 0

    with serial.Serial('/dev/rfcomm0', 115200, timeout=100) as ser:

        # Once we've opened the serial port, do a first alignment and
        # initialize last_realtime from the hardware clock.

        synced, a_raw = sync_to_uq(ser)
        last_realtime = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)

        # Outer loop captures until process is terminated
        while True:

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

                # print("DEBUG: time_delta: " + str(time_delta))

                for i in range(time_delta - 1):

                    # Increment the total count of data points, including dropped data.
                    capture_count += 1

                    for idx in range(3):

                        accel[idx].append(math.nan)
                        w_vel[idx].append(math.nan)
                        mag_angle[idx].append(math.nan)

            # Increment the total count of data points by one for our new good data point.
            capture_count += 1

            for idx, value in enumerate([a_1_int, a_2_int, a_3_int]):
                accel[idx].append(value / 32768.0 * 16)

            for idx, value in enumerate([w_1_int, w_2_int, w_3_int]):
                w_vel[idx].append(value / 32768.0 * 2000)

            for idx, value in enumerate([angle_1_int, angle_2_int, angle_3_int]):
                mag_angle[idx].append(value / 32768.0 * 180)

            if capture_count >= capture_window:

                capture_count = 0
                window_dict = {'accel': accel,
                               'w_vel': w_vel,
                               'mag_angle': mag_angle,
                               'timestamp': time.clock_gettime(time.CLOCK_MONOTONIC_RAW),
                               'desync_count': desync_count}

                # Uncomment this to print the raw x-axis accel values for debugging.
                # print(accel[0])

                # Clear the temporary measurement buffers.

                accel = [[] for i in range(3)]
                w_vel = [[] for i in range(3)]
                mag_angle = [[] for i in range(3)]
                queue.put(window_dict)


def detect_events(sample_freq, queue):
    """
    Does FFT based event detection and plotting.

    :type sample_freq: int
    :param sample_freq: Sampling frequency in Hz
    :type queue: Multiprocessing Queue
    :param queue: detect_events() reads from this queue to acquire batches of IMU data captured by the acquire_data()
                  process.
    :return: None
    """

    BAR_WIDTH = 10
    BAR_SPACING = 15
    EDGE_OFFSET = 3
    GRAPH_SIZE = (500, 100)
    DATA_SIZE = (500, 100)

    graph = sg.Graph(GRAPH_SIZE, (0, 0), DATA_SIZE)

    # detect_events() blocks on the queue read, so data processing rate is driven by the rate that

    layout = [[sg.Image(filename='blank.png', key='image')],
              [graph],
              [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - Scratching Detection',
                       location=(800, 400))

    window.Layout(layout).Finalize()

    while True:

        imu_data_dict = queue.get()

        # Check the timestamp on this data frame. If it's more than one second behind the current
        # CLOCK_MONOTONIC_RAW time, then detect_events() isn't keeping up with the incoming stream
        # of IMU measurements comming from acquire_data(). Jump back to the top of the while loop,
        # and get another data frame. We'll keep doing that until we're reading fresh data again.

        # TODO: the threshold here might need to be adjusted, and perhaps the whole queue should be cleared.

        if (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - imu_data_dict['timestamp']) > 1:

            continue

        df = pd.DataFrame(data=imu_data_dict['w_vel'][0], columns=['x'])

        # print("Number of NaN points: " + str(df.isna().sum()))

        # Doing a linear interpolation that replaces the NaN placeholders for desynced, dropped data
        # with a straight line between the last good data point before a dropout and the next good data
        # point. This does seem to improve the signal-to-noise ratio in the FFT.
        df = df.interpolate()

        yf = fftpack.rfft(df.loc[:, 'x'])

        # Is this a better way of getting the power spectrum than numpy.abs(yf?)
        yf = numpy.sqrt((yf * yf.conj()).real)

        xf = fftpack.rfftfreq((len(df.index)), 1./sample_freq)

        peaks, peaks_dict = signal.find_peaks(yf, height=1)

        #print(xf[peaks], yf[peaks])
        #print(peaks, peaks_dict)

        # Check if any peak has been detected between 3 and 5 Hz, along the x axis. This is our rule for detecting
        # tooth-brushing behavior.

        event, values = window.Read(timeout=0, timeout_key='timeout')

        graph.Erase()

        for i in range(len(peaks_dict['peak_heights'])):
            graph_value = peaks_dict['peak_heights'][i]
            graph.DrawRectangle(top_left=(i * BAR_SPACING + EDGE_OFFSET, graph_value),
                                bottom_right=(i * BAR_SPACING + EDGE_OFFSET + BAR_WIDTH, 0), fill_color='blue')
            graph.DrawText(text=graph_value, location=(i * BAR_SPACING + EDGE_OFFSET + 25, graph_value + 10))

        if event == 'Exit' or event is None:
            sys.exit(0)

        # Scratching motion of puppet causes slow roll ( < 1 hz) along x axis.
        if numpy.where(peaks_dict['peak_heights'] > 800)[0].size > 0:
            window.FindElement('image').Update(filename='rat.png')
            print("Scratching Detected!")
        else:
            window.FindElement('image').Update(filename='blank.png')


def main():
    """
    Main function acquires data from the BWT901CL for a fixed length of time, and then performs a FFT analysis on it.

    :return: None
    """

    default_sample_freq = 10
    capture_window = 20

    data_queue = Queue()

    # Spawn acquire_data() in a separate process, so that IMU data acquisition won't be delayed by the data processing
    # in detect_events(). acquire_data() and detect_events() are the producer and consumer in a producer-consumer
    # design pattern.

    acquire_data_p = Process(target=acquire_data, args=(default_sample_freq, capture_window, data_queue,))
    acquire_data_p.daemon = True
    acquire_data_p.start()

    detect_events(default_sample_freq, data_queue)

if __name__ == "__main__":
    main()
