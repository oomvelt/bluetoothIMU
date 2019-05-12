
import bluetooth
import subprocess

target_name = "HC-06"
target_address = None

nearby_devices = bluetooth.discover_devices()

for bdaddr in nearby_devices:
    if target_name == bluetooth.lookup_name(bdaddr):
        target_address = bdaddr
        break

if target_address is not None:
    print("found target bluetooth device with address ", target_address)
else:
    print("could not find target bluetooth device nearby")
    exit()

# Above code snippet taken from:
# https://people.csail.mit.edu/albert/bluez-intro/c212.html#pbz-choosing-device

ret = subprocess.call(["rfcomm", "bind", "0", str(target_address)])

print("ret: ", ret)

ret = subprocess.call(["chmod", "0766", "/dev/rfcomm0"])

print("ret: ", ret)