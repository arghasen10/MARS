#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[14]:


u1 = 0.99
u2 = np.array([0.95353177, 0.97413376])
u3 = np.array([0.94790384, 0.93329807, 0.93489855])
u4 = np.array([0.90470352, 0.91364137, 0.911636  , 0.93256851, 0.88701471])
u5 = np.array([0.84170317, 0.90405693, 0.91812147, 0.9261504 , 0.85211643])


# In[19]:


u1_mean = np.mean(u1)
u2_mean = np.mean(u2)
u3_mean = np.mean(u3)
u4_mean = np.mean(u4)
u5_mean = np.mean(u5)


# Calculate the standard deviation
u1_std = np.std(u1)
u2_std = np.std(u2)
u3_std = np.std(u3)
u4_std = np.std(u4)
u5_std = np.std(u5)


# Define labels, positions, bar heights and error bar heights
labels = ['u1', 'u2', 'u3', 'u4', 'u5']
x_pos = np.arange(len(labels))
CTEs = [u1_mean, u2_mean, u3_mean, u4_mean, u5_mean]
error = [u1_std, u2_std, u3_std, u4_std, u5_std]


# In[24]:


fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, width=0.4, align='center', ecolor='black', capsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.set_xlabel('Number of Users')
ax.set_ylabel('Weighted F1-Score')
# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




