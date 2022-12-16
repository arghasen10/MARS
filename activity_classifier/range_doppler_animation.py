import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


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
            flag=1
        mmwave_df = mmwave_df.append(d, ignore_index=True)
        last_date = d['timenow']

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: '2022-10-14 ' + ':'.join(e.split('_')))
    mmwave_df['datetime'] = pd.to_datetime(mmwave_df['datetime'])
    mmwave_df = mmwave_df[['datetime', 'rangeIdx', 'dopplerIdx', 'x_coord', 'y_coord', 'rp_y', 'doppz']]
    print(last_date)
    return mmwave_df.dropna()


# In[3]:


mmwave_data = pd.concat([process_mmWave(f) for f in glob.glob("static*.txt")])


# In[4]:


cropped_data = mmwave_data.copy()


# In[5]:


# cropped_data = cropped_data[cropped_data.datetime > pd.to_datetime('2022-10-14 21:56:24')]
# cropped_data = cropped_data[cropped_data.datetime < pd.to_datetime('2022-10-14 21:56:27')] 


# In[6]:


df = cropped_data.copy()


# In[7]:


doppz = np.array(df['doppz'].values.tolist())
rangeIdx = np.array(df['rangeIdx'].values.tolist())
dopplerIdx = np.array(df['dopplerIdx'].values.tolist())
doppz = (doppz - doppz.min()) / (doppz.max() - doppz.min())
static_frame = doppz[0]
fig, axs = plt.subplots(1, 2, figsize=(9, 7), gridspec_kw={'width_ratios': [8, 1]})


# In[ ]:


for i in range(doppz.shape[0]):
    for j in range(len(rangeIdx[i])):
        if dopplerIdx[i][j] != 0:
            dopp_frame = static_frame
            try:
                dopp_frame[:, rangeIdx[i][j]-5:rangeIdx[i][j]+5] = doppz[i, :, rangeIdx[i][j]-5:rangeIdx[i][j]+5]
            except:
                try:
                    dopp_frame[:, rangeIdx[i][j]:rangeIdx[i][j]+5] = doppz[i, :, rangeIdx[i][j]:rangeIdx[i][j]+5]
                except:
                    dopp_frame[:, rangeIdx[i][j]-5:rangeIdx[i][j]] = doppz[i, :, rangeIdx[i][j]-5:rangeIdx[i][j]]
            axs[0].cla()
            axs[1].cla()
            title = "Object Number = "+ str(j)
            axs[0].set_title(title)
            yticks = np.linspace(doppz.shape[1], 0, 5, dtype=np.int)
            xticks = np.linspace(doppz.shape[2], 0, 5, dtype=np.int)
            yticklabel = np.linspace(doppz.shape[1]/2, -doppz.shape[1]/2, 5, dtype=np.int)
            xticklabel = np.linspace(doppz.shape[2], -doppz.shape[2]/2, 5, dtype=np.int)
            sns.heatmap(dopp_frame, ax=axs[0], cbar_ax=axs[1], vmax=1, vmin=0)
            axs[0].set_xticks(xticks)
            axs[0].set_yticks(yticks)
            axs[0].set_xticklabels(xticklabel)
            axs[0].set_yticklabels(yticklabel)
            fig.canvas.draw()
            fig.show()

# In[9]:

plt.show()
doppz.shape


# In[34]:


data = [json.loads(val) for val in open(glob.glob("static*.txt")[0], "r")]
for d in data:
    print(d['dopplerIdx'])


# In[14]:


fig1, axs1 = plt.subplots(1, 2, figsize=(9, 7), gridspec_kw={'width_ratios': [8, 1]})
sns.heatmap(doppz[0], ax=axs1[0], cbar_ax=axs1[1], vmax=1, vmin=0)
fig1.show()


# In[15]:





# In[ ]:




