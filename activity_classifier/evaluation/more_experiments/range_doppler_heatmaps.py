#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


df = pd.read_pickle('macro_df.pkl')

activities = list(df.Activity.value_counts().to_dict().keys())
df1 = df[df.Position == 1]
for act in activities:
    print(act)
    fig, ax = plt.subplots()
    df2 = df1[df1.Activity == act]
    doppz = np.array(df['doppz'].values.tolist())
    data = doppz.std(axis=0)
    data = (data - data.min()) / (data.max() - data.min())
    sns.heatmap(data, cbar_ax=None)
    ax.set_xlabel('Range bins')
    ax.set_ylabel('Doppler bins')
    ax.set_yticks(np.arange(0, 15, 4))
    ax.set_yticklabels(np.arange(-8, 8, 4))
    ax.set_xticks(np.arange(0, 255, 16))
    ax.set_xticklabels(np.arange(1, 256, 16))
    break


# In[47]:


df = pd.read_pickle('micro_df2.pkl')
activities = list(df.Activity.value_counts().to_dict().keys())
df1 = df[df.Position == 1]
for act in activities:
    print(act)
    fig, ax = plt.subplots()
    df2 = df1[df1.Activity == act]
    doppz = np.array(df['doppz'].values.tolist())
    data = doppz.std(axis=0)
    data = (data - data.min()) / (data.max() - data.min())
    sns.heatmap(data, cbar_ax=None)
    ax.set_xlabel('Range bins')
    ax.set_ylabel('Doppler bins')
    ax.set_yticks(np.arange(0, 127, 16))
    ax.set_yticklabels(np.arange(-64, 64, 16))
    ax.set_xticks(np.arange(0, 63, 8))
    ax.set_xticklabels(np.arange(1, 64, 8))
    plt.margins(0)
    break


# In[ ]:




