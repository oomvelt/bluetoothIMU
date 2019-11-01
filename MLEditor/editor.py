import base64
import PySimpleGUI as sg
import sys
import zipfile
import json
import natsort


def fixBadZipfile(zipFile):  
     f = open(zipFile, 'r+b')  
     data = f.read()  
     pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature  
     if (pos > 0):  
         self._log("Truncating file at location " + str(pos + 22) + ".")  
         f.seek(pos + 22)   # size of 'ZIP end of central directory record' 
         f.truncate()  
         f.close()  
     else: 
         print("ERROR: Could not fix zipfile...")


image_list = []

archive = zipfile.ZipFile('log_11-01-2019-14-02-32.zip', 'r')

file_list = []

for this_file in archive.namelist():
    file_list.append(this_file)

image_count = 0

#TODO: check that there is only one json file in the zip archive; squawk and stop otherwise.

data_dict_file = next(filename for filename in file_list if 'logger' in filename)
print(data_dict_file)

data_dict = json.load(archive.open(data_dict_file))

print(data_dict['w_vel'][0])

print("Loading images...")
for filename in natsort.natsorted(file_list):
    if 'image' in filename:
        this_file = archive.open(filename)
        this_str = base64.b64encode(this_file.read())
        image_list.append(this_str)
        image_count += 1

archive.close()

graph = sg.Graph(canvas_size=(400, 200), graph_bottom_left=(-52, -52), graph_top_right=(52, 52),
                     background_color='white', key='chart', float_values=True)   

layout = [[sg.Image(data=image_list[0], key='image')],
          [graph],
          [sg.Button('Back', size=(10,1), font='Helvetica 14'),
           sg.Slider(range=(0,image_count - 1),
           default_value=0,
           size=(20,15),
           orientation='horizontal',
           font=('Helvetica', 12),
           enable_events=True,
           key='slider'),
           sg.Button('Forward', size=(10,1), font='Helvetica 14')],
          [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]

    # create the window and show it without the plot
window = sg.Window('Data Labelling', location=(800, 400))

window.Layout(layout).Finalize()

print(data_dict['w_vel'][0])

for idx in range(len(data_dict['w_vel'][0])):
    y = (data_dict['w_vel'][0][idx]/10)
    graph.DrawCircle((idx-100, y), 1, line_color='blue', fill_color='blue')

while True:

    event, values = window.Read()
    if event == 'Exit' or event is None:
        print("Exiting...")
        sys.exit(0)

    if event == 'slider':
        window.Element('image').update(data=image_list[int(values['slider'])])

    if event == 'Back':
        #Decrement the index of the slider and the image array, but don't decrement below 0.)
        index = max(values['slider'] - 1, 0)
        window.FindElement('slider').Update(index)
        window.Element('image').update(data=image_list[int(values['slider'])])

    if event == 'Forward':
        #Increment the index of the slider and the image array, but don't increment above the size of the image array.)
        index = min(values['slider'] + 1, image_count - 1)
        window.FindElement('slider').Update(index)
        window.Element('image').update(data=image_list[int(values['slider'])])


