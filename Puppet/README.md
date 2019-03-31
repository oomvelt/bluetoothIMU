# Puppet Demo with GUI



## Version .1

This code demonstrates Fast Fourier Transform based realtime detection of scratching behavior when using a hand puppet. There is a command line based version, streaming_puppet.py, and a version with a full GUI, streaming_puppet_gui.py.

## Installation

Dependencies:

This demo is implemented in Python 3 and requires the Python 3 serial (pySerial), numpy, pandas, scipy, and pysimplegui libraries in order to run. Excepting pysimplegui, all of these can be downloaded as part of the Anaconda Python Data Analytics distribution, [here](https://www.anaconda.com/download/#linux) (use the Python 3.x download link):

pysimplegui must be installed with pip install.

Note: the other libraries can alternatuvely be installed individually with pip install, as well.


## Quick Start

If the BWT901CL has already been paired with the Linux system, follow the instructions directly below. If the BWT901CL hasn't yet been paired, please follow the instructions under "Not Quick Start" to pair the device first, and then return to follow these steps.

The following must be done once each time the Linux system is booted up:

1. From the Linux command prompt, type 'sudo rfcomm bind 0 XX:YY:ZZ:AA:BB:CC' (where XX:YY:ZZ:AA:BB:CC is the device address for the BWT901CL).
2. Type 'sudo chmod 766 /dev/rfcomm0'. The Bluetooth bound serial port should now be ready for use by streaming_puppet.py/streaming_puppet_gui.py.

The following must be performed each time the demo is to be run:

No command line arguments are required. Navigate to the "Puppet" directory at the command line, and start either streaming_puppet.py or streaming_puppet_gui.py by itself, using the Anaconda pyhton 3. If anaconda is installed in its default location, the command with look something like "~/anaconda3/bin/python3 streaming_puppet_gui.py."

Note: streaming_puppet_gui.py needs to png files, rat.png and blank.png, to be located in the same directory. 

rfcomm_serial.py looks for motion along the x-axis of the accelerometer. So hold the accelerometer with the x-axis pointing either directly towards you,
or directly away from you.


## Not Quick STart

Bluetooth setup for rfcomm_serial.py:

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

## License
[BSD-3](https://opensource.org/licenses/BSD-3-Clause)
