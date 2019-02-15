#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import csv
import os
import imageio
from random import randint
from scipy import ndimage

import keras
from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# In[2]:

def get_rand_transform(img):
    angle = randint(-15, 15)
    img = ndimage.rotate(img, angle, reshape = False)
    return img

def load_data(dir):
    
    ntc_file = dir + '/numbers_to_classes.csv'
    ntc_dict = {}
    with open(ntc_file) as ntcf:
        reader = csv.DictReader(ntcf, delimiter=',')
        for row in reader:
            ntc_dict[int(row['class_number'])] = row['sign_class']

    train_file = dir + '/gt_train.csv'
    train_dict = {}
    test_dict = {}
    test_file = dir + '/gt_test.csv'

    x_train_len = 0
    x_test_len = 0
    train_num = 70000
    test_num = 10000
    
    X_train = np.zeros(shape=(train_num, 48, 48, 3))
    Y_train = np.zeros(shape=(train_num,))
    X_test = np.zeros(shape=(test_num, 48, 48, 3))
    Y_test = np.zeros(shape=(test_num,))
    
    with open(train_file) as file:
        cnt = 0
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            train_dict[row['filename']] = int(row['class_number'])
            img = imageio.imread(dir + "/train/"+row['filename'])
            img_rot = get_rand_transform(img)
            class_num = int(row['class_number'])
            if x_train_len<train_num-1:
                X_train[x_train_len] = img
                Y_train[x_train_len] = class_num
                X_train[x_train_len+1] = img_rot
                Y_train[x_train_len+1] = class_num
                x_train_len += 2
            else:
                pass
            cnt += 2
            if x_train_len == train_num-1:
                break
            if cnt % 1000 == 0:
                print("Load {} train images".format(cnt))
        X_train = X_train[:x_train_len]
        Y_train = Y_train[:x_train_len]
    print("Train dictionary size: {}".format(len(train_dict)))

    with open(test_file) as file:
        cnt = 0
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            if x_test_len<test_num-1:
                test_dict[row['filename']] = int(row['class_number'])
                img = imageio.imread(dir + "/test/"+row['filename'])
                class_num = int(row['class_number'])
                X_test[x_test_len] = img
                Y_test[x_test_len] = class_num
                x_test_len += 1
                if x_test_len % 1000 == 0:
                    print("Load {} test images".format(x_test_len))

        X_test = X_test[:x_test_len]
        Y_test = Y_test[:x_test_len]
        print("Test dictionary size: {}".format(len(test_dict)))

    classes = len(ntc_dict)
    
    return X_train, Y_train, X_test, Y_test, classes


# In[3]:


def NormalizeImg(img_set):
    norm = np.zeros(img_set.shape, np.float)
    for i in range(len(img_set)):
        img = img_set[i].astype(dtype=float)
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        norm[i] = img
    return norm


# In[4]:


X_train, Y_train, X_test, Y_test, classes = load_data("rtsd-r1")

X_train = NormalizeImg(X_train)
X_test = NormalizeImg(X_test)

Y_train = tf.keras.utils.to_categorical(Y_train, classes)
Y_test = tf.keras.utils.to_categorical(Y_test, classes)


# In[5]:


model = Sequential()

model.add(Conv2D(100, kernel_size=(7, 7), activation='relu', input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(150, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(250, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))


# In[8]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=1)


# In[ ]:


model.save_weights('checkpoint/model.h5')
model.save('checkpoint/signs.h5')
print('Saved')

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
