# pyBWT901CL



## Version .2

pyBWT901CL demonstrates various aspects of capturing data from the BWT901CL Bluetooth 9 Axis IMU, and performing event detection on that data.

fft_demo.py demonstrates FFT based detection of tooth brushing behavior on static data read from a file.

rfcomm_serial.py demonstrates acquisition of live data from the BWT901CL, followed by application of the same FFT based detection technique used in fft_demo.py.



## Installation

Dependencies:

pyBWT901CL is implemented in Python 3 and requires the Python 3 serial (pySerial), numpy, pandas, scipy, and matplotlib libraries in order to run. All of these can be downloaded as part of the Anaconda Python Data Analytics distribution, [here](https://www.anaconda.com/download/#linux) (use the Python 3.x download link):

Or installed individually with pipinstall.

A. Setup for fft_demo.py

fft_demo.py requires data from the UCI "Dataset for ADL Recognition with Wrist-worn Accelerometer Data." This can be downloaded [here.](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer) Place the HMP_Dataset folder at the same folder level as fft_demo.py.


B. Bluetooth setup for rfcomm_serial.py:

The BWT901CL must first be paired with your Linux system. This should only need to be done one time, for any given BWT901CL device.

1. Make sure the BWT901CL is switched on, and either has battery charge or is plugged in.
2. Open a Linux command line terminal.
3. Type: 'sudo bluetoothctl'
4. From within the bluetoothctl interactive interface, type:
5. 'agent on'
6. 'default-agent'
7. 'scan on'
8. bluetoothctl should now begin to scan for Bluetooth devices. Look for an entry ending with "HC-06".
   the identifier before that, in the pattern XX:YY:ZZ:AA:BB:CC, is the Bluetooth device address of the
   BWT901CL. Record this address on paper or in a text file. Enter the following:
9. 'pair XX:YY:ZZ:AA:BB:CC' (where XX:YY:ZZ:AA:BB:CC is the device address for the BWT901CL)
10. bluetoothctl will ask for a PIN or passkey. For the BWT901CL, the PIN/passkey is:
11. '1234'
12. bluetoothcl should indicate "Paired: yes; Pairing successful." It may also report "Connected: no". This is expected.
13. Type 'exit' to leave bluetoothctl.

The following must be done once each time the Linux system is booted up:

1. From the Linux command prompt, type 'sudo rfcomm bind 0 XX:YY:ZZ:AA:BB:CC' (where XX:YY:ZZ:AA:BB:CC is the device address for the BWT901CL).
2. Type 'sudo chmod 766 /dev/rfcomm0' note: if rfcomm0 already existed prior to step 14, it may be necessary to bind to a Bluetooth port other than '0'.
   If so, update the serial port address near the top of the acquire_data() function in rfcomm_serial.py
3. Update the bluetooth device address in the 
3. The Bluetooth bound serial port should now be ready for use by rfcomm_serial.py



## Usage

No command line arguments are required. Start either fft_demo.py or rfcomm_serial.py by itself from the command line or a file browser.

rfcomm_serial.py looks for motion along the x-axis of the accelerometer. So hold the accelerometer with the x-axis pointing either directly towards you,
or directly away from you.



## License
[BSD-3](https://opensource.org/licenses/BSD-3-Clause)
