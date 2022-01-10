#!/usr/bin/env python
# coding: utf-8

"""
CNN classification using Keras
dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

#install cv2 and tensorflow package and use the package to establish the deep learning model
import glob
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# down load the data and divide it into training test and validation set
path = './chest_xray/'

# define paths
train_normal_dir = path + 'train/NORMAL/'
train_pneu_dir = path + 'train/PNEUMONIA/'

test_normal_dir = path + 'test/NORMAL/'
test_pneu_dir = path + 'test/PNEUMONIA/'

val_normal_dir = path + 'val/NORMAL/'
val_pneu_dir = path + 'val/PNEUMONIA/'


# find all files, our files have extension jpeg
train_normal_cases = glob.glob(train_normal_dir + '*jpeg')
train_pneu_cases = glob.glob(train_pneu_dir + '*jpeg')

test_normal_cases = glob.glob(test_normal_dir + '*jpeg')
test_pneu_cases = glob.glob(test_pneu_dir + '*jpeg')

val_normal_cases = glob.glob(val_normal_dir + '*jpeg')
val_pneu_cases = glob.glob(val_pneu_dir + '*jpeg')


# make path using / instead of \\ ... this may be a redudant step
train_normal_cases = [x.replace('\\', '/') for x in train_normal_cases]
train_pneu_cases = [x.replace('\\', '/') for x in train_pneu_cases]
test_normal_cases = [x.replace('\\', '/') for x in test_normal_cases]
test_pneu_cases = [x.replace('\\', '/') for x in test_pneu_cases]
val_normal_cases = [x.replace('\\', '/') for x in val_normal_cases]
val_pneu_cases = [x.replace('\\', '/') for x in val_pneu_cases]

## Prepare data
# balance the training set and validation set
def set_balancer(train_list, val_list, portion):
    if len(train_list)/len(val_list) > portion:
        rn.shuffle(train_list)
        balanced_train_list = train_list[: (len(train_list)//portion*(portion-1))]
        balanced_val_list = train_list[(len(train_list)//portion*(portion-1)):] + val_list
    return balanced_train_list, balanced_val_list

train_pneu_cases, val_pneu_cases = set_balancer(train_pneu_cases, val_pneu_cases, 5)
train_normal_cases, val_normal_cases = set_balancer(train_normal_cases, val_normal_cases, 6)

# load image and pre-process it by grayscaling and resizing 
def read_img(img_path):
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img,(64,64))
    grayscale = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    return grayscale

# create lists for train, test & validation cases, create labels as well
train_list = []
test_list = []
val_list = []

for x in train_normal_cases:
    x = read_img(x)
    train_list.append([x, 0])

for x in test_normal_cases:
    x = read_img(x)
    test_list.append([x, 0])

for x in train_pneu_cases:
    x = read_img(x)
    train_list.append([x, 1])

for x in test_pneu_cases:
    x = read_img(x)
    test_list.append([x, 1])

for x in val_normal_cases:
    x = read_img(x)
    val_list.append([x, 0])

for x in val_pneu_cases:
    x = read_img(x)
    val_list.append([x, 1])

# shuffle/randomize data as they were loaded in order: normal cases, then pneumonia cases
rn.shuffle(train_list)
rn.shuffle(test_list)
rn.shuffle(val_list)

# create dataframes
train_df = pd.DataFrame(train_list, columns=['image', 'label'])
test_df = pd.DataFrame(test_list, columns=['image', 'label'])
val_df = pd.DataFrame(val_list, columns=['image', 'label'])

# save processed images so as to directly load them next time
train_list = np.array(train_list)
test_list = np.array(test_list)
val_list = np.array(val_list)
np.save('train_list.npy', train_list)
np.save('test_list', test_list)
np.save('val_list', val_list)

# After you saved the dataset above, execute from here next time
# load the saved images
train_list = np.load('train_list.npy', allow_pickle=True)
test_list = np.load('test_list.npy', allow_pickle=True)
val_list = np.load('val_list.npy', allow_pickle=True)

# Shuffle the data
idx = np.arange(test_list.shape[0])
np.random.shuffle(idx)
test_list = test_list[idx]

idx = np.arange(train_list.shape[0])
np.random.shuffle(idx)
train_list = train_list[idx]

idx = np.arange(val_list.shape[0])
np.random.shuffle(idx)
val_list = val_list[idx]

"""
1. Normalize the training and testing dataset
2. One hot encoding the classes
"""

# normalize
train_list_norm = [i[0]/255 for i in train_list]
test_list_norm = [i[0]/255 for i in test_list]
val_list_norm = [i[0]/255 for i in val_list]

# dummy step. Make sure it's numpy array
train_list = np.asarray(train_list)
test_list = np.asarray(test_list)
val_list = np.asarray(val_list)


from keras.utils import np_utils
# one-hot encoding
y_train = [i[1] for i in train_list]
y_test = [i[1] for i in test_list]
y_val = [i[1] for i in val_list]
y_train_onehot = np_utils.to_categorical(y_train, num_classes=2)
y_test_onehot = np_utils.to_categorical(y_test, num_classes=2)
y_val_onehot = np_utils.to_categorical(y_val, num_classes=2)


## Training Model : CNN-model

# model architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same', input_shape=(64,64, 1), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same', input_shape=(64,64, 1), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same', input_shape=(64,64, 1), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 
model.add(Dense(1024, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax')) #2 classes

# print the summary of the model
model.summary()

# training

# dummy step. Confirm the data type of datasets
train_list_norm = np.asarray(train_list_norm).astype('float32')
test_list_norm = np.asarray(test_list_norm).astype('float32')
val_list_norm = np.asarray(val_list_norm).astype('float32')

# hyper-parameters of the model
from keras.callbacks import EarlyStopping,ModelCheckpoint
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
earlyStopping = EarlyStopping(monitor='val_loss', patience=2)
modelCheckpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# start training
history = model.fit(train_list_norm, y_train_onehot, validation_split=0.3, validation_data=val_list_norm, 
                    epochs=10, batch_size=32, verbose=1,
                   callbacks=[earlyStopping, modelCheckpoint])

# save the trained model
model.save('model2.h5')

# load the saved trained model
from keras.models import load_model
model = load_model('model2.h5')

# use test set to determine performance of the model
model.evaluate(test_list_norm, y_test_onehot)

# generate model performance report
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(test_list_norm), axis=-1)
print(classification_report(y_test, y_pred))

# print confusion matrix of the model
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                  index=["Actual Normal", "Actual Pneumonia"],
                  columns=["Predicted Normal", "Predicted Pneumonia"])
# print it
cm

# plot the history of loss and accuracy during training
def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')
plt.subplot(1,2,2)
plot_train_history(history, 'accuracy','val_accuracy')