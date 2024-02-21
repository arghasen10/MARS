#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
tf.random.set_seed(32)
np.random.seed(32)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import seaborn as sns


# In[2]:


def scale(doppz, Max=36240, Min=9343):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=5):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


# In[3]:


def get_train_test(df1):
    doppz = np.array(df1['doppz'].values.tolist())
    label = df1['Activity'].values
    dop_max, dop_min = doppz.max(), doppz.min()
    frame_stack=1
    doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, dop_max, dop_min), label, frame_stack)
    lbl_map ={'Clapping': 0,
          'jumping': 1, 
          'lunges': 2, 
          'squats': 3,
          'waving' : 4}


    X_norm = doppz_scaled_stacked
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e],new_labels))), num_classes=5)
    
    return train_test_split(X_norm, y, test_size=0.3, random_state=42)


# In[4]:


def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(2,5),(1,2),padding="same",activation='relu',input_shape=(16,256,1)),
        tf.keras.layers.Conv2D(64,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32,"relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5,"softmax")
    ])
    return model


# In[5]:


def get_result(X_test1, y_test1):
    model2 = get_model()
    model2.load_weights('macro_weights_distance1.h5')
    pred = model2.predict([X_test1])
    conf_matrix = confusion_matrix(np.argmax(y_test1, axis=1),np.argmax(pred, axis=1))
    total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
    total = np.round(total,2)
    labels = ['Clapping', 'jumping', 'lunges', 'squats', 'waving']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.show()
    cls_rep = classification_report(np.argmax(y_test1, axis=1), np.argmax(pred, axis=1))
    print(cls_rep)
    return conf_matrix, cls_rep


# In[6]:


def get_overall_X_train_y_train(df_1, df_2, df_3):
    X_train_1, X_test_1, y_train_1, y_test_1 = get_train_test(df_1)
    X_train_2, X_test_2, y_train_2, y_test_2 = get_train_test(df_2)
    X_train_3, X_test_3, y_train_3, y_test_3 = get_train_test(df_3)
    X_train = np.concatenate([X_train_1, X_train_2, X_train_3])
    X_test = np.concatenate([X_test_1, X_test_2, X_test_3])
    y_train = np.concatenate([y_train_1, y_train_2, y_train_3])
    y_test = np.concatenate([y_test_1, y_test_2, y_test_3])
    return X_train, X_test, y_train, y_test, X_test_1, y_test_1, X_test_2, y_test_2, X_test_3, y_test_3


# In[7]:


loc="macro_df.pkl"
frame_stack=5
df = pd.read_pickle(loc)

df = df[df.Activity != 'walking']
df = df[df.Activity != 'running']

df_1 = df[df.Position == 1]
df_2 = df[df.Position == 2]
df_3 = df[df.Position == 3]


# In[8]:


X_train, X_test, y_train, y_test, X_test_1, y_test_1, X_test_2, y_test_2, X_test_3, y_test_3 = get_overall_X_train_y_train(df_1, df_2, df_3)


# In[9]:


model=get_model()
print(model.summary())
# model.load_weights('macro_weights.h5')
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
best_save=tf.keras.callbacks.ModelCheckpoint(filepath='macro_weights_distance1.h5',save_weights_only=True, 
                                             monitor='val_accuracy',mode='max',save_best_only=True)
tbd=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folder}')

model.fit(
    X_train,
    y_train,
    epochs=500,
    validation_split=0.2,
    batch_size=32,
    callbacks=[best_save,tbd]
)


# In[10]:


weighted_orientation_activity = []
cfm_1, rep_1 = get_result(X_test_1, y_test_1)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_1.split('\n')[2:-5]]])
wf1_1 = float(rep_1.split('\n')[-2].strip().split()[-2])


# In[11]:


cfm_2, rep_2 = get_result(X_test_2, y_test_2)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_2.split('\n')[2:-5]]])
wf1_2 = float(rep_2.split('\n')[-2].strip().split()[-2])


# In[12]:


cfm_3, rep_3 = get_result(X_test_3, y_test_3)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_3.split('\n')[2:-5]]])
wf1_3 = float(rep_3.split('\n')[-2].strip().split()[-2])


# In[13]:


weighted_orientation_activity


# In[20]:


width = 0.2
x = np.arange(5)
# plot data in grouped manner of bar type
plt.bar(x-0.2, weighted_orientation_activity[0], width)
plt.bar(x, weighted_orientation_activity[1], width)
plt.bar(x+0.2, weighted_orientation_activity[2], width)
plt.xticks(x, ['Clapping', 'jumping', 'lunges', 'squats', 'waving'])
plt.xlabel("Activity")
plt.ylabel("F1-Score")
plt.ylim(0.8,1.03)
plt.legend(["2m", "3m", "5m"], ncols=3)
plt.show()


# In[ ]:




