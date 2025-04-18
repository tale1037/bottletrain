import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import os, random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import cm

import glob

full = []
label1 = []
for img in glob.glob("./dataset/water/*.jpeg") + glob.glob(
        "./dataset/water/*.png"):
    n = cv2.imread(img)
    n = Image.fromarray(n, 'RGB')
    n_res = n.resize((256, 256))
    full.append(n_res)
    label1.append(0)

empty = []
label2 = []
for img in glob.glob("./dataset/empty/*.jpeg") + glob.glob("./dataset/empty/*.jpg"):
    m = cv2.imread(img)
    m = Image.fromarray(m, 'RGB')
    m_res = m.resize((256, 256))
    m1 = m_res.rotate(15)
    m2 = m_res.rotate(-15)
    empty.append(m_res)
    empty.append(m1)
    empty.append(m2)
    label2.append(1)
    label2.append(1)
    label2.append(1)


img=full+empty
label=label1+label2

img = np.array(img)
label= np.array(label)

plt.figure(figsize=(14,5))
x, y = 4, 2
for i in range(8):
    r = np.random.randint(0 , img.shape[0] , 1)
    plt.subplot(y, x, i+1)
    plt.imshow(img[r[0]], cmap='gray')
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.title(label[r[0]],color="w")
    plt.xticks([]) , plt.yticks([])
plt.show()

imgs=[]
from tensorflow.keras.utils import img_to_array
for i in img:
  x = img_to_array(i)
  imgs.append(x)
imgs = np.array(imgs)/255
imgs.shape

datagen = ImageDataGenerator(height_shift_range=0.3,fill_mode='nearest')
datagen.fit(imgs)
generated_imgs = []
generated_labels = []
for x_batch, y_batch in datagen.flow(imgs, label, batch_size=1):
    generated_imgs.extend(x_batch)
    generated_labels.extend(y_batch)
    if len(generated_imgs) >= imgs.shape[0]:
        break
imgs = np.concatenate((imgs, generated_imgs))
label = np.concatenate((label, generated_labels))

datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             rotation_range=10,fill_mode='nearest')
datagen.fit(imgs)
generated_imgs = []
generated_labels = []
for x_batch, y_batch in datagen.flow(imgs, label, batch_size=1):
    generated_imgs.extend(x_batch)
    generated_labels.extend(y_batch)
    if len(generated_imgs) >= imgs.shape[0]:
        break
imgs = np.concatenate((imgs, generated_imgs))
label = np.concatenate((label, generated_labels))

imgs.shape,label.shape

from sklearn.model_selection import train_test_split
label_cat=to_categorical(label,3)
x_train , x_test , y_train , y_test = train_test_split(imgs, label,
                                            test_size = 0.25,
                                            random_state = 42)

np.unique(y_train,return_counts=True)

np.unique(y_test,return_counts=True)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

num_classes = 2
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

epochs = 30
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256,256, 3)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
#es=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=5)
history=model.fit(x_train, y_train,epochs=epochs,verbose=1,validation_data=(x_test, y_test),class_weight=class_weight_dict)
score = model.evaluate(x_test, y_test, verbose=1)

plt.plot(history.history['accuracy'], label='acc', color='red')
plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

y_pred = model.predict(x_test)
ax = sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1)), cmap="binary",annot=True,fmt="d")

model.save("./savedmodels/Modified_CNN_model/saved_model.h5")