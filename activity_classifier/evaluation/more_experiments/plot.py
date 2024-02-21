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
    Max = doppz.max()
    Min = doppz.min()
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=10):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


def get_aug_pipe(frame_stack=10):
    Input = tf.keras.layers.Input(shape=(16, 256, frame_stack))
    net = tf.keras.layers.Cropping2D(((0, 0), (0, 24)), name='Crop')(Input)
    net = tf.keras.layers.Resizing(height=48, width=48, name='Resize_48x48')(net)
    net = tf.keras.layers.RandomTranslation(height_factor=(0.0, 0.0),
                                            width_factor=(0.0, 0.8), fill_mode='wrap', name='R_shift')(net)
    pipe = tf.keras.Model(inputs=[Input], outputs=[net], name='Aug_pipe')
    return pipe


# In[3]:


def run_aug_once(doppz_scaled_stacked, pipe):
    return pipe.predict(doppz_scaled_stacked, batch_size=256)


# In[4]:


class Dataset:
    def __init__(self, loc="merged_df2.pkl", class_count=3080, frame_stack=10, dop_min=9343,
                 dop_max=36240):
        # Temp vals
        print(f"loading dataset from {loc}")
        df = pd.read_pickle(loc)
        df = df[df.Activity != '  '].reset_index()
        doppz = np.array(df['doppz'].values.tolist())
        label = df['Activity'].values
        doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, dop_max, dop_min), label, frame_stack)

        # class members
        self.class_count = class_count
        self.do_num_aug = np.ceil(self.class_count / df['Activity'].value_counts()).to_dict()
        self.pipe = get_aug_pipe(frame_stack=frame_stack)
        self.data, self.label = self.process(doppz_scaled_stacked, new_labels)

    def augument(self, stacked_doppz_sub_arr, num_aug=None):
        total_arr = np.concatenate([run_aug_once(stacked_doppz_sub_arr, self.pipe) for _ in range(num_aug)], axis=0)
        final_indices = np.random.choice(np.arange(0, total_arr.shape[0]), size=self.class_count, replace=True)
        return total_arr[final_indices]

    def process(self, doppz_scaled_stacked, new_labels):
        data = []
        lbl = []
        for activ, num_aug in self.do_num_aug.items():
            print(f"on activity {activ} -> augument for {num_aug} times")
            data.append(self.augument(doppz_scaled_stacked[new_labels == activ], int(num_aug)))
            lbl.extend([activ] * self.class_count)
        data = np.concatenate(data, axis=0)
        lbl = np.array(lbl)
        return data, lbl


# In[5]:


def get_dataset():
    data = Dataset(loc='merged_df2.pkl',
                   class_count=3080,
                   frame_stack=10,
                   dop_min=9343,dop_max=36240)
    lbl_map ={'Clapping': 0,
              'jumping': 1, 
              'lunges': 2, 
              'running': 3,
              'squats': 4,
              'walking' : 5,
              'waving' : 6}

    X_norm = data.data
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], data.label))), num_classes=7)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


# In[6]:


def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(5,5),(1,1),padding="same",activation='relu',input_shape=(48,48,10)),
        tf.keras.layers.MaxPool2D((3,3)),
        tf.keras.layers.Conv2D(64,(3,3),(1,1),padding="same",activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(96,(3,3),(1,1),padding="same",activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32,"relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(7,"softmax")
    ])
    return model


# In[ ]:


X_train, X_test, y_train, y_test=get_dataset()
model=get_model()
model.load_weights('saved_weights_macro_more.h5')
#model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

#folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
#best_save=tf.keras.callbacks.ModelCheckpoint(filepath='saved_weights_macro_more.h5',save_weights_only=True,
#                                                monitor='val_accuracy',mode='max',save_best_only=True)
#tbd=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folder}')

#model.fit(
#    X_train,
#    y_train,
#    epochs=500,
#    validation_split=0.2,
#    batch_size=32,
#    callbacks=[best_save,tbd])


# In[ ]:


pred=model.predict([X_test])


# In[ ]:


conf_matrix = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred, axis=1))
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['Clapping', 'jumping', 'lunges', 'running', 'squats', 'walking', 'waving']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.show()


# In[ ]:


print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))


# In[ ]:




