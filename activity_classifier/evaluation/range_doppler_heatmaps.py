#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 32})
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# In[46]:


df = pd.read_pickle('../macro_df.pkl')

activities = list(df.Activity.value_counts().to_dict().keys())
df1 = df[df.Position == 1]
for act in activities:
    print(act)
    fig, ax = plt.subplots()
    df2 = df1[df1.Activity == act]
    doppz = np.array(df2['doppz'].values.tolist())
    data = doppz.std(axis=0)
    data = (data - data.min()) / (data.max() - data.min())
    sns.heatmap(data, cbar_ax=None)
    ax.set_xlabel('Range bins')
    ax.set_ylabel('Doppler bins')
    ax.set_ylim(-0.25,16.25)
    ax.set_yticks(np.arange(0, 17, 8))
    ax.set_yticklabels(np.arange(-8, 9, 8))
    ax.set_xticks(np.arange(0, 257, 64))
    ax.set_xticklabels(np.arange(0, 257, 64), rotation=0)
    plt.tight_layout()
    plt.savefig(act+'.eps')
    plt.show()



# In[47]:


df = pd.read_pickle('../micro_df2.pkl')
activities = list(df.Activity.value_counts().to_dict().keys())
df1 = df[df.Position == 1]
for act in activities:
    print(act)
    fig, ax = plt.subplots()
    if act == 'eating-food':
        df1 = df[df.Position == 2]
        df2 = df1[df1.Activity == act]
        doppz = np.array(df2['doppz'].values.tolist())
    else:
        df2 = df1[df1.Activity == act]
        doppz = np.array(df2['doppz'].values.tolist())
    data = doppz.std(axis=0)
    data = (data - data.min()) / (data.max() - data.min())
    sns.heatmap(data, cbar_ax=None)
    ax.set_xlabel('Range bins')
    ax.set_ylabel('Doppler bins')
    ax.set_yticks(np.arange(0, 129, 64))
    ax.set_yticklabels(np.arange(-64, 65, 64))
    ax.set_xticks(np.arange(0, 65, 16))
    ax.set_xticklabels(np.arange(0, 65, 16), rotation=0)
    plt.tight_layout()
    plt.savefig(act + '.eps')
    plt.show()


# In[ ]:




