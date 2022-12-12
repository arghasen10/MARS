import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.grid(alpha=0.2)

col_names = ["Date", "Time", "numObj", "rangeIdx", "range", "dopplerIdx", "doppler", "peakVal", "x", "y", "z",
            'magicNumber', 'version', 'totalPacketLen', 'platform', 'frameNumber', 'timeCpuCycles', 'numDetectedObj',
            'numTLVs', 'subFrameNumber', 'tlv_type', 'tlv_length', 'tlv_numObj', 'tlv_xyzQFormat']


maxNumObj = -1
maxlength = -1


def max_numObj(file, maxNumObj):
    df = pd.read_csv(file)
    noObjs = df['numDetectedObj'].to_numpy()
    for noObj in noObjs:
        noObj = int(noObj)
        if noObj > maxNumObj:
            maxNumObj = noObj
            print('maxNumObj', maxNumObj)
    return maxNumObj


def find_files_in_path(old_path):
    files = []
    # old_path = 'data_collection/day2Argha/'
    all_files = os.listdir(old_path)
    for file in all_files:
        filename = file.split('.')
        if filename[-1] == 'json':
            files.append(old_path + file)
    return files


def process_json_to_df(files):
    dfs = []
    for file in files:
        data = [json.loads(line) for line in open(file, 'r')]
        output = pd.DataFrame()
        for d in data:
            activity = {'activity': str(file.split('.')[0].split('/')[-1])}
            d.update(activity)
            output = output.append(d, ignore_index=True)

        output = output[col_names]
        dfs.append(output)
    return dfs


def process_txt_to_csv(files):
    for file in files:
        filepath = file.split('.')[0]
        filepath += '.csv'
        with open(filepath, 'w') as f:
            csv.DictWriter(f, fieldnames=col_names).writeheader()
        with open(file, 'r') as datafile:
            lines = datafile.readlines()
            for line in lines:
                data = json.loads(line)
                activity = {'activity': str(file.split('.')[0])}
                data.update(activity)
                with open(filepath, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                    # writer.writeheader()
                    writer.writerow(data)
        print('done writing file', filepath)


def plot_range_vs_frame(df):
    range_vals = []
    time_vals = []
    range_arr = df['range'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('Range vs frame')
    plt.xlabel('Frames')
    plt.ylabel('Range')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_range'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


def plot_rangeIdx_vs_frame(df):
    range_vals = []
    time_vals = []
    range_arr = df['rangeIdx'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('rangeIdx vs frame')
    plt.xlabel('Frames')
    plt.ylabel('rangeIdx')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_rangeIdx'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


def plot_dopplerIdx_vs_frame(file):
    range_vals = []
    time_vals = []
    range_arr = df['dopplerIdx'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('dopplerIdx vs frame')
    plt.xlabel('Frames')
    plt.ylabel('dopplerIdx')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_peakVal'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


def plot_peakVal_vs_frame(file):
    range_vals = []
    time_vals = []
    range_arr = df['peakVal'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('peakVal vs frame')
    plt.xlabel('Frames')
    plt.ylabel('peakVal')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_peakVal'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


def plot_x_coord_vs_frame(file):
    range_vals = []
    time_vals = []
    range_arr = df['x'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('x_coord vs frame')
    plt.xlabel('Frames')
    plt.ylabel('x_coord')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_x_coord'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


def plot_y_coord_vs_frame(file):
    range_vals = []
    time_vals = []
    range_arr = df['y'].to_numpy()
    counter = 0
    for data in range_arr:
        counter += 1
        for d in data:
            range_vals.append(d)
            time_vals.append(counter)
    plt.title('y_coord vs frame')
    plt.xlabel('Frames')
    plt.ylabel('y_coord')
    plt.scatter(time_vals, range_vals)
    plotfile = 'single_user_random2'
    plotfile += '_y_coord'
    plotfile += '.pdf'
    plt.savefig(plotfile, bbox_inches='tight', format='pdf')
    plt.show()


# def zero_padding(df):
#     rangeIdxarr = df['rangeIdx'].to_numpy()
#     print('rangeIdxarr.shape', rangeIdxarr.shape)
#     dopplerIdxarr = df['dopplerIdx'].to_numpy()
#     rangearr = df['range'].to_numpy()
#     peakValarr = df['peakVal'].to_numpy()
#     x_coordarr = df['x_coord'].to_numpy()
#     y_coordarr = df['y_coord'].to_numpy()
#     output = pd.DataFrame()
#     col_name = ['datenow', 'timenow', 'rangeIdx', 'dopplerIdx', 'numDetectedObj', 'range', 'peakVal', 'x_coord',
#                 'y_coord', 'rp_y', 'noiserp_y', 'azimuthz', 'doppz', 'activity']
#     print('len(y_coordarr)', len(y_coordarr))
#     print('y_coordarr.shape', y_coordarr.shape)
#     for e in range(0, len(y_coordarr)):
#         t = 20 - len(rangeIdxarr[e])
#         datenow = df['datenow'][e]
#         timenow = df['timenow'][e]
#         rangeIdxarre = np.pad(rangeIdxarr[e], pad_width=(0, t), mode='constant')
#         t = 20 - len(dopplerIdxarr[e])
#         dopplerIdxarre = np.pad(dopplerIdxarr[e], pad_width=(0, t), mode='constant')
#         numDetectedObj = df['numDetectedObj'][e]
#         t = 20 - len(rangearr[e])
#         rangearre = np.pad(rangearr[e], pad_width=(0, t), mode='constant')
#         t = 20 - len(peakValarr[e])
#         peakValarre = np.pad(peakValarr[e], pad_width=(0, t), mode='constant')
#         t = 20 - len(x_coordarr[e])
#         x_coordarre = np.pad(x_coordarr[e], pad_width=(0, t), mode='constant')
#         t = 20 - len(y_coordarr[e])
#         y_coordarre = np.pad(y_coordarr[e], pad_width=(0, t), mode='constant')
#         rp_ye = df['rp_y'][e]
#         noiserp_ye = df['noiserp_y'][e]
#         azimuthze = df['azimuthz'][e]
#         print('azimuthze.shape', azimuthze.shape)
#         doppze = df['doppz'][e]
#         activitye = df['activity'][e]

#         dicts = {'datenow': datenow, 'timenow': timenow, 'rangeIdx': rangeIdxarre, 'dopplerIdx': dopplerIdxarre,
#                  'numDetectedObj': numDetectedObj, 'range': rangearre, 'peakVal': peakValarre, 'x_coord': x_coordarre,
#                  'y_coord': y_coordarre, 'rp_y': rp_ye, 'noiserp_y': noiserp_ye, 'azimuthz': azimuthze, 'doppz': doppze,
#                  'activity': activitye}
#         output = output.append(dicts, ignore_index=True)
#     output = output[col_name]
#     return output


# files = find_files_in_path('dataset/')
dfs = process_json_to_df(['dataset/20220720_162220.json'])

# dfs = pd.read_csv('/home/argha/guicomposer/runtime/gcruntime.v6/mmWave_Demo_Visualizer/data_collection/rpi_data/')
print('len(dfs)', len(dfs))

rangearrval = []
countarr = []
for df in dfs:
    plot_x_coord_vs_frame(df)
    plot_y_coord_vs_frame(df)
    plot_rangeIdx_vs_frame(df)
    plot_dopplerIdx_vs_frame(df)
    plot_range_vs_frame(df)
    plot_peakVal_vs_frame(df)
