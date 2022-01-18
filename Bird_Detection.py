import PIL
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


dir = r'D:\Dataset\New folder'
cat = ["Bird", "Not_Bird"]

print("[INFO] loading images...")

data = []
labels = []

for c in cat:
    path = os.path.join(dir, c)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(244,244))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        data.append(image)
        labels.append(c)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

data = np.array(data, dtype='float32')
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels)

print(trainX.shape)
print(testX.shape)

baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(244, 244, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print('[INFO] Model Compiling...')
opt = Adam(lr=1e-4, decay=1e-4/10)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(trainX, trainY, batch_size=32, epochs=10)

pred = model.predict(testX, batch_size=10)

pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")
