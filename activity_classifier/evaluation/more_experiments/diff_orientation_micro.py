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


def scale(doppz, Max=9343, Min=36240):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=10):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


# In[3]:


def get_train_test(df1):
    doppz = np.array(df1['doppz'].values.tolist())
    label = df1['Activity'].values
    dop_max, dop_min = doppz.max(), doppz.min()
    doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, dop_max, dop_min), label, frame_stack)
    lbl_map ={'laptop-typing':0, 
              'sitting':1, 
              'phone-typing':2, 
              'phone-talking':3, 
              'playing-guitar':4, 
              'eating-food':5
              }

    X_norm = doppz_scaled_stacked

    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], new_labels))), num_classes=6)

    return train_test_split(X_norm, y, test_size=0.3, random_state=42)


# In[4]:


def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(3,2),(2,1),padding="same",activation='relu',input_shape=(128,64,2)),
        tf.keras.layers.Conv2D(64,(3,3),(2,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(3,3),(2,2),padding="same",activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32,"relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(6,"softmax")
    ])
    return model


# In[5]:


def get_result(X_test1, y_test1):
    model2 = get_model()
    model2.load_weights('micro_weights_orientation.h5')
    pred = model2.predict([X_test1])
    conf_matrix = confusion_matrix(np.argmax(y_test1, axis=1),np.argmax(pred, axis=1))
    total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
    total = np.round(total,2)
    labels = ['laptop-typing','sitting', 'phone-typing', 'phone-talking', 'playing-guitar', 'eating-food']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.show()
    cls_rep = classification_report(np.argmax(y_test1, axis=1), np.argmax(pred, axis=1))
    print(cls_rep)
    return conf_matrix, cls_rep


# In[6]:


def get_overall_X_train_y_train(df_front, df_left, df_back, df_right):
    X_train_f, X_test_f, y_train_f, y_test_f = get_train_test(df_front)
    X_train_l, X_test_l, y_train_l, y_test_l = get_train_test(df_left)
    X_train_b, X_test_b, y_train_b, y_test_b = get_train_test(df_back)
    X_train_r, X_test_r, y_train_r, y_test_r = get_train_test(df_right)
    X_train = np.concatenate([X_train_f, X_train_l, X_train_b, X_train_r])
    X_test = np.concatenate([X_test_f, X_test_l, X_test_b, X_test_r])
    y_train = np.concatenate([y_train_f, y_train_l, y_train_b, y_train_r])
    y_test = np.concatenate([y_test_f, y_test_l, y_test_b, y_test_r])
    return X_train, X_test, y_train, y_test, X_test_f, y_test_f, X_test_l, y_test_l, X_test_b, y_test_b, X_test_r, y_test_r    


# In[7]:


loc="micro_df2.pkl"
frame_stack=2
df = pd.read_pickle(loc)
df = df[df.Activity != '  '].reset_index()

df_front = df[df.Orientation == 'Front']
df_back = df[df.Orientation == 'back']
df_left = df[df.Orientation == 'left']
df_right = df[df.Orientation == 'right']


# In[8]:


X_train, X_test, y_train, y_test, X_test_f, y_test_f, X_test_l, y_test_l, X_test_b, y_test_b, X_test_r, y_test_r = get_overall_X_train_y_train(df_front, df_left, df_back, df_right)


# In[9]:


# model = get_model()
# model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

# folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
# best_save=tf.keras.callbacks.ModelCheckpoint(filepath='micro_weights_orientation.h5',save_weights_only=True, 
#                                              monitor='val_accuracy',mode='max',save_best_only=True)
# tbd=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folder}')

# model.fit(
#     X_train,
#     y_train,
#     epochs=500,
#     validation_split=0.2,
#     batch_size=32,
#     callbacks=[best_save,tbd]
# )


# In[10]:


weighted_orientation_activity = []
cfm_front, rep_front = get_result(X_test_f, y_test_f)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_front.split('\n')[2:-5]]])
wf1_front = float(rep_front.split('\n')[-2].strip().split()[-2])


# In[11]:


cfm_left, rep_left = get_result(X_test_l, y_test_l)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_left.split('\n')[2:-5]]])
wf1_left = float(rep_left.split('\n')[-2].strip().split()[-2])


# In[12]:


cfm_back, rep_back = get_result(X_test_b, y_test_b)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_back.split('\n')[2:-5]]])
wf1_back = float(rep_back.split('\n')[-2].strip().split()[-2])


# In[13]:


cfm_right, rep_right = get_result(X_test_r, y_test_r)
weighted_orientation_activity.append([float(f.split()[3]) for f in [e.strip() for e in rep_right.split('\n')[2:-5]]])
wf1_right = float(rep_right.split('\n')[-2].strip().split()[-2])


# In[14]:


weighted_orientation_activity


# In[16]:


width = 0.2
x = np.arange(6)
# plot data in grouped manner of bar type
plt.bar(x-0.3, weighted_orientation_activity[0], width)
plt.bar(x-0.1, weighted_orientation_activity[1], width)
plt.bar(x+0.1, weighted_orientation_activity[2], width)
plt.bar(x+0.3, weighted_orientation_activity[3], width)
plt.xticks(x, ['laptop\ntyping','sitting', 'phone\ntyping', 'phone\ntalking', 'playing\nguitar', 'eating\nfood'])
plt.xlabel("Activity")
plt.ylabel("F1-Score")
plt.legend(["Front", "Left", "Back", "Right"], ncols=4)
plt.show()


# In[ ]:




