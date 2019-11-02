import base64
import PySimpleGUI as sg
import sys
import zipfile
import json
import natsort
import math


def updateFigure(graph, marker_line, data_dict, camera_index, imu_min_value):


    #values.index(min(values))

    min_imu = min(data_dict['imu_timestamp'], key=lambda this_imu_timestamp: abs(this_imu_timestamp - data_dict['camera_timestamp'][camera_index]))
    min_imu_index = data_dict['imu_timestamp'].index(min_imu)
    graph.RelocateFigure(marker_line, min_imu_index, imu_min_value)
    


def main():

    image_list = []

    # TODO: Load zip archive based on text box and load button, or some such.
 
    #archive = zipfile.ZipFile('log_11-02-2019-14-32-43.zip', 'r')
    archive = zipfile.ZipFile('log_11-02-2019-15-00-36.zip', 'r')

    file_list = []

    for this_file in archive.namelist():
        file_list.append(this_file)

    image_count = 0

    #TODO: check that there is only one json file in the zip archive; squawk and stop otherwise.

    data_dict_file = next(filename for filename in file_list if 'logger' in filename)

    data_dict = json.load(archive.open(data_dict_file))

    print("Loading images...")
    for filename in natsort.natsorted(file_list):
        if 'image' in filename:
            this_file = archive.open(filename)
            this_str = base64.b64encode(this_file.read())
            image_list.append(this_str)
            image_count += 1

    archive.close()

    imu_graph_range = len(data_dict['w_vel'][0])
    imu_min_value = min(data_dict['w_vel'][0])
    imu_max_value = max(data_dict['w_vel'][0])

    # Need to coerce the min and max values away from 0.0, or the graph draw functions
    # have divide by zero issues.
    if imu_min_value == 0.0:
        imu_min_value = -100

    if imu_max_value == 0.0:
        imu_max_value = 100

    imu_graph = sg.Graph(canvas_size=(600, imu_graph_range), graph_bottom_left=(0, imu_min_value), graph_top_right=(imu_graph_range, imu_max_value),
                         background_color='white', key='chart', float_values=True)   

    layout = [[sg.Image(data=image_list[0], key='image')],
              [imu_graph],
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
    
    # Plot the IMU data.
    for idx in range(len(data_dict['w_vel'][0])):
        y = (data_dict['w_vel'][0][idx])    
        imu_graph.DrawCircle((idx, y), 1, line_color='blue', fill_color='blue')


    marker_line = imu_graph.DrawLine((0,imu_min_value), (0,imu_max_value), color="yellow", width=2)
    


    while True:

        event, values = window.Read()
        if event == 'Exit' or event is None:
            print("Exiting...")
            sys.exit(0)

        if event == 'slider':
            window.Element('image').update(data=image_list[int(values['slider'])])
            updateFigure(imu_graph, marker_line, data_dict, int(values['slider']), imu_min_value)

        if event == 'Back':
            #Decrement the index of the slider and the image array, but don't decrement below 0.)
            index = int(max(values['slider'] - 1, 0))
            window.FindElement('slider').Update(index)
            window.Element('image').update(data=image_list[index])
            updateFigure(imu_graph, marker_line, data_dict, index, imu_min_value)

        if event == 'Forward':
            #Increment the index of the slider and the image array, but don't increment above the size of the image array.)
            index = int(min(values['slider'] + 1, image_count - 1))
            window.FindElement('slider').Update(index)
            window.Element('image').update(data=image_list[index])
            updateFigure(imu_graph, marker_line, data_dict, index, imu_min_value)


if __name__ == "__main__":
    main()
