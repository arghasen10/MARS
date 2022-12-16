import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({'font.size': 32})
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def reformat_milli(df):
    df['datetime'] = [ts + f'_{i}' for ts, e in df.groupby('datetime').count().iloc[:, 0].to_dict().items() for i in
                      range(e)]
    return df


def process_mmWave(filename):
    flag = 0
    data = [json.loads(val) for val in open(filename, "r")]
    mmwave_df = pd.DataFrame()
    last_date = ''
    for d in data:
        if flag == 0:
            print(d['timenow'])
            flag = 1
        mmwave_df = mmwave_df.append(d, ignore_index=True)
        last_date = d['timenow']

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: '2022-10-14 ' + ':'.join(e.split('_')))
    mmwave_df['datetime'] = pd.to_datetime(mmwave_df['datetime'])
    mmwave_df = mmwave_df[['datetime', 'x_coord', 'y_coord', 'rp_y', 'doppz']]
    print(last_date)
    return mmwave_df.dropna()


mmwave_data = pd.concat([process_mmWave(f) for f in glob.glob("*.txt")])
cropped_data = mmwave_data.copy()
cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:07:30')]
cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:07:34')]

df = cropped_data.copy()
doppz = np.array(df['doppz'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [8, 1]})

yticks = np.linspace(16, 0, 4, dtype=np.int)
xticks = np.linspace(256, 0, 8, dtype=np.int)
g = sns.heatmap(doppz[0], ax=axs[0], cbar_ax=axs[1], vmax=1, vmin   =0)
g.set_xlabel('Range bins')
g.set_ylabel('Doppler bins')
g.set_yticks(np.arange(0, 17, 8))
g.set_yticklabels(np.arange(-8, 9, 8))
g.set_xticks(np.arange(0, 257, 64))
g.set_xticklabels(np.arange(0, 257, 64), rotation=0)
# g.set_xticks(xticks)
# g.set_yticks(yticks)
# g.set_xticklabels(xticks)
# g.set_yticklabels(yticks)
plt.tight_layout()
plt.savefig('plot_static.eps')

cropped_data = mmwave_data.copy()
cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:19:00')]
cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:19:02')]

df = cropped_data.copy()
doppz = np.array(df['doppz'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [8, 1]})

yticks = np.linspace(16, 0, 4, dtype=np.int)
xticks = np.linspace(256, 0, 8, dtype=np.int)
g = sns.heatmap(doppz[0], ax=axs[0], cbar_ax=axs[1], vmax=1, vmin=0)
g.set_xlabel('Range bins')
g.set_ylabel('Doppler bins')
g.set_yticks(np.arange(0, 17, 8))
g.set_yticklabels(np.arange(-8, 9, 8))
g.set_xticks(np.arange(0, 257, 64))
g.set_xticklabels(np.arange(0, 257, 64), rotation=0)
# g.set_xticks(xticks)
# g.set_yticks(yticks)
# g.set_xticklabels(xticks)
# g.set_yticklabels(yticks)
plt.tight_layout()
plt.savefig('plot_dynamic.eps')


cropped_data = mmwave_data.copy()
cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:43:57')]
cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:43:59')]

df = cropped_data.copy()
doppz = np.array(df['doppz'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [8, 1]})

yticks = np.linspace(16, 0, 4, dtype=np.int)
xticks = np.linspace(256, 0, 8, dtype=np.int)
g = sns.heatmap(doppz[-1], ax=axs[0], cbar_ax=axs[1], vmax=1, vmin=0)
g.set_xlabel('Range bins')
g.set_ylabel('Doppler bins')
g.set_yticks(np.arange(0, 17, 8))
g.set_yticklabels(np.arange(-8, 9, 8))
g.set_xticks(np.arange(0, 257, 64))
g.set_xticklabels(np.arange(0, 257, 64), rotation=0)
# g.set_xticks(xticks)
# g.set_yticks(yticks)
# g.set_xticklabels(xticks)
# g.set_yticklabels(yticks)
plt.tight_layout()
plt.savefig('plot_macro_micro.eps')

cropped_data = mmwave_data.copy()
cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:46:35')]
cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:46:38')]

df = cropped_data.copy()
doppz = np.array(df['doppz'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [8, 1]})

yticks = np.linspace(128, 0, 8, dtype=np.int)
xticks = np.linspace(64, 0, 8, dtype=np.int)
g = sns.heatmap(doppz[3], ax=axs[0], cbar_ax=axs[1], vmax=1, vmin=0)
g.set_xlabel('Range bins')
g.set_ylabel('Doppler bins')
g.set_yticks(np.arange(0, 129, 64))
g.set_yticklabels(np.arange(-64, 65, 64))
g.set_xticks(np.arange(0, 65, 16))
g.set_xticklabels(np.arange(0, 65, 16), rotation=0)
# g.set_xticks(xticks)
# g.set_yticks(yticks)
# g.set_xticklabels(xticks)
# g.set_yticklabels(yticks)
plt.tight_layout()
plt.savefig('plot_micro_macro.eps')


cropped_data = mmwave_data.copy()
cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:56:24')]
cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:56:27')]

df = cropped_data.copy()
doppz = np.array(df['doppz'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [8, 1]})

yticks = np.linspace(16, 0, 4, dtype=np.int)
xticks = np.linspace(256, 0, 8, dtype=np.int)
g = sns.heatmap(doppz[-1], ax=axs[0], cbar_ax=axs[1], vmax=1, vmin=0)
g.set_xlabel('Range bins')
g.set_ylabel('Doppler bins')
g.set_yticks(np.arange(0, 17, 8))
g.set_yticklabels(np.arange(-8, 9, 8))
g.set_xticks(np.arange(0, 257, 64))
g.set_xticklabels(np.arange(0, 257, 64), rotation=0)
# g.set_xticks(xticks)
# g.set_yticks(yticks)
# g.set_xticklabels(xticks)
# g.set_yticklabels(yticks)
plt.tight_layout()
plt.savefig('plot_zombie.eps')
