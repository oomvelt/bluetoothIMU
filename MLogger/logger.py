from multiprocessing import Process, Queue
import collections
import queue
import time
import serial
import numpy
import pandas as pd
from scipy import signal
from scipy import fftpack
import math
import sys
import PySimpleGUI as sg
import pygame
import pygame.camera
from pygame.locals import *
import io
import base64
import json
import datetime
import zipfile
import os


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
    imu_timestamp = []
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
                    imu_timestamp.append(realtime)

                    for idx in range(3):

                        accel[idx].append(math.nan)
                        w_vel[idx].append(math.nan)
                        mag_angle[idx].append(math.nan)

            # Increment the total count of data points by one for our new good data point.
            capture_count += 1
            imu_timestamp.append(realtime)

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
                               'imu_timestamp': imu_timestamp,
                               'desync_count': desync_count}

                # Uncomment this to print the raw x-axis accel values for debugging.
                # print(accel[0])
                # print(window_dict['imu_timestamp'])
                
                queue.put(window_dict)

                # Clear the temporary measurement buffers.
                accel = [[] for i in range(3)]
                w_vel = [[] for i in range(3)]
                mag_angle = [[] for i in range(3)]
                imu_timestamp = []
                

def log_data(imu_queue):
    """

    :type imu_queue: Multiprocessing Queue
    :param imu_queue: log_data() reads from this queue to acquire batches of IMU data captured by the acquire_data()
                  process.
    :return: None
    """
    
    pygame.init()
    pygame.camera.init()

    cam = pygame.camera.Camera("/dev/video0", (640, 480))
    cam.start()
    
    image_array = []
    camera_timestamp_array = []
    image_capture_count = 0
    
    chart = collections.deque(201*[0], 201)

    graph = sg.Graph(canvas_size=(400, 200), graph_bottom_left=(-52, -52), graph_top_right=(52, 52),
                     background_color='white', key='chart', float_values=True)

    layout = [[sg.Image(key='image', filename='oomvelt.png')],
              [graph, sg.Multiline(key='notes', default_text='', size=(45, 5), do_not_clear=True)],
              [sg.Button('Log', size=(10, 1), font='Helvetica 14')],
              [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Logger',
                       location=(800, 400))

    window.Layout(layout).Finalize()

    accel = [[] for i in range(3)]
    w_vel = [[] for i in range(3)]
    mag_angle = [[] for i in range(3)]
    imu_timestamp = []
    desync_count = 0

    logger_dict = {'accel': accel,
                   'w_vel': w_vel,
                   'mag_angle': mag_angle,
                   'imu_timestamp': imu_timestamp,
                   'desync_count': desync_count}

    log_state = False
    # GUI update loop
    logging_done = False
    loop_count = 0
    while not logging_done:
        loop_count += 1
           
        done_reading_IMU_queue = False
       
        while not done_reading_IMU_queue:

            try: 
                latest_imu_data_dict = imu_queue.get(False)

                # Check the timestamp on this data frame. If it's more than one second behind the current
                # CLOCK_MONOTONIC_RAW time, then detect_events() isn't keeping up with the incoming stream
                # of IMU measurements coming from acquire_data(). Jump back to the top of the while loop,
                # and get another data frame. We'll keep doing that until we're reading fresh data again.

                # TODO: the threshold here might need to be adjusted, and perhaps the whole queue should be cleared

                if (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - latest_imu_data_dict['imu_timestamp'][-1]) > 1:
                    print("WARNING: stale data detected. Skipping data...")
                    continue

                # If we are in the logging state, append the last batch of collected logger data to the logger dict.

                if log_state:
                    
                    for item in latest_imu_data_dict.keys():
                        
                        # Only desync_count isn't a 1-d or 2-d array
                        if item == 'desync_count':            
                            logger_dict[item] = latest_imu_data_dict[item]
                            
                        else:
                            len(latest_imu_data_dict[item])
                            
                            # Check if this is a 2-d array. If it is, extend each subarray in turn with new data.
                            if isinstance(latest_imu_data_dict[item][0], list):
                                for idx in range(len(latest_imu_data_dict[item])):
                                    logger_dict[item][idx].extend(latest_imu_data_dict[item][idx])
                                    
                            else:        
                                logger_dict[item].extend(latest_imu_data_dict[item])
                            
                else:
                    
                    # Add the latest IMU element to the stripchart's data structure.
                    for item in latest_imu_data_dict['w_vel'][0]:
                        chart.appendleft(item)
                        chart.pop()
                    
                    graph.Erase()
                    for x in range(0, 200):
                        y = (chart[x]/10)
                        graph.DrawCircle((x-100, y), 1, line_color='blue', fill_color='blue')
            # If the queue throws Empty, we've read everything and can continue on.
            except queue.Empty:
                done_reading_IMU_queue = True       
        
        if log_state:
            this_image = cam.get_image()
            camera_timestamp_array.append(time.clock_gettime(time.CLOCK_MONOTONIC_RAW))
            image_array.append(this_image)
            image_capture_count += 1
           
        else:
            this_image = cam.get_image()
            # Note: I couldn't find a way to get PySImpleGUI's Image class to accept a pygame binary string,
            # no matter how I massaged it. As a workaround, I am currently saving each image out as a .png file
            # to /dev/shm, which should be ramdisk, and loading it in below. This does involve OS calls, which are
            # likely slowing things down. Only used during the pre-logging phase, though.
            pygame.image.save(this_image, "/dev/shm/tmp.png")

        if image_capture_count >= 400:
            logging_done = True
            print("INFO: logging done. Archiving...")
            logger_dict['camera_timestamp'] = camera_timestamp_array       
            log_timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m-%d-%Y-%H-%M-%S")
            with open('logger_dict_' + log_timestamp + '.json', 'w') as fp:
                json.dump(logger_dict, fp)
            archive = zipfile.ZipFile("log_" + log_timestamp + ".zip", 'w')
            archive.write("logger_dict_" + log_timestamp + ".json")
            os.remove("logger_dict_" + log_timestamp + ".json")
            for idx, item in enumerate(image_array):
                image_file = str("image" + str(idx) + ".png")
                pygame.image.save(item, image_file)               
                archive.write(image_file)
                os.remove(image_file)
            archive.close()
            print("INFO: Archiving done.")

        event, values = window.Read(timeout=0, timeout_key='timeout')

        if log_state == False:
            window.Element('image').Update(filename="/dev/shm/tmp.png")
            logger_dict['notes'] = window.Element('notes').Get()

        if event == 'Exit' or event is None:
            cam.stop()
            sys.exit(0)
        elif event == 'Log':
            log_state = True


def main():
    """
    Main function acquires data from the BWT901CL for a fixed length of time, and then performs a FFT analysis on it.

    :return: None
    """

    default_sample_freq = 10
    capture_window = 20
    image_capture_window = 700

    IMU_data_queue = Queue()

    # Spawn acquire_data() in a separate process, so that IMU data acquisition won't be delayed by the data processing
    # in detect_events(). acquire_data() and detect_events() are the producer and consumer in a producer-consumer
    # design pattern.

    acquire_data_p = Process(target=acquire_data, args=(default_sample_freq, capture_window, IMU_data_queue,))
    acquire_data_p.daemon = True
    acquire_data_p.start()

    log_data(IMU_data_queue)


if __name__ == "__main__":
    main()
