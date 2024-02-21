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


def scale(doppz, Min=9343, Max=36240):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=10):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


# In[3]:


class Dataset:
    def __init__(self, loc="macro_df.pkl", frame_stack=5, lbl_map=None):
        print(f"loading dataset from {loc}")
        df = pd.read_pickle(loc)
        self.frame_stack = frame_stack
        self.df = df[df.Activity != '  '].reset_index()
        self.lbl_map=lbl_map
        doppz = np.array(self.df['doppz'].values.tolist())
        self.dop_max, self.dop_min = doppz.max(), doppz.min()
        
    
    def activity_stack(self, df, dop_max, dop_min):
        s_t = 0
        c=0
        X_train_data = []
        y_train_data = []
        X_test_data = []
        y_test_data = []
        df['ts'] = df['datetime'].apply(lambda e: pd.to_datetime(e).value/10**9)
        gaps = (np.abs(df['ts'].values[1:]-df['ts'].values[0:-1])>10).nonzero()[0]+1
        for g in gaps:
            df_new = df.iloc[s_t:g]
            s_t = g+1
            doppz = np.array(df_new['doppz'].values.tolist())
            label = df_new['Activity'].values
            X_norm = doppz
            y = to_categorical(np.array(list(map(lambda e: self.lbl_map[e], label))), num_classes=7)
            if X_norm.shape[0] < 11:
                continue
            if X_norm.shape[0] < 100:
                X_train, y_train = StackFrames(scale(X_norm, dop_min, dop_max), y, self.frame_stack)
                X_train_data.append(X_train)
                y_train_data.append(y_train)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, shuffle=False)

                X_train, y_train = StackFrames(scale(X_train, dop_min, dop_max), y_train, self.frame_stack)
                X_train_data.append(X_train)
                y_train_data.append(y_train)
                X_test, y_test = StackFrames(scale(X_test, dop_min, dop_max), y_test, self.frame_stack)
                X_test_data.append(X_test)
                y_test_data.append(y_test)
        return (np.concatenate(X_train_data),
                np.concatenate(X_test_data),
                np.concatenate(y_train_data),
                np.concatenate(y_test_data))
    def process(self):
        return self.activity_stack(self.df, self.dop_max, self.dop_min)
    
    def __del__(self):
        del self.df


# In[4]:


def get_dataset():
    lbl_map ={'Clapping': 0,
      'jumping': 1, 
      'lunges': 2, 
      'running': 3,
      'squats': 4,
      'walking' : 5,
      'waving' : 6}
    X_train, X_test, y_train, y_test = Dataset(loc='macro_df.pkl', frame_stack=1,lbl_map=lbl_map).process()#5
    return X_train, X_test, y_train, y_test


# In[5]:


def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(2,5),(1,2),padding="same",activation='relu',input_shape=(16,256,1)),
        tf.keras.layers.Conv2D(64,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(2,3),(1,2),padding="same",activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32,"relu"),
#         tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(7,"softmax")
    ])
    return model


# In[6]:


X_train, X_test, y_train, y_test=get_dataset()


# In[7]:


model=get_model()


# In[8]:


# model.load_weights('macro_weights.h5')


# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
best_save=tf.keras.callbacks.ModelCheckpoint(filepath='macro_weights.h5',save_weights_only=True, 
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


# In[11]:


pred = model.predict([X_test])
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred, axis=1))
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['Clapping', 'jumping', 'lunges', 'running', 'squats', 'walking', 'waving']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.show()


# In[12]:


print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))


# In[3]:


import pandas as pd
df = pd.read_pickle('macro_df.pkl')


# In[4]:


import numpy as np


# In[5]:


df['ts'] = df['datetime'].apply(lambda e: pd.to_datetime(e).value/10**9)

gaps = (np.abs(df['ts'].values[1:]-df['ts'].values[0:-1])>10).nonzero()[0]+1


# In[11]:





# In[15]:





# In[ ]:




