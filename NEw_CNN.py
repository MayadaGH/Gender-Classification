# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import pandas as pd 
#import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(32,32,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) 
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu')) # fully connected layer 
classifier.add(Dense(units=1,activation='sigmoid')) # 2 class

import tensorflow
adam = tensorflow.keras.optimizers.Adam(lr=0.001)
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])





from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
#
#
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size)

print("[INFO] describing images...")
df = pd.read_csv("F:/Level 3/Semester 2/Pattern/wiki/wiki/dataset.csv")
rawImages = []
features = []
labels = [] 

for i in range(0,50000):
    image = cv2.imread('F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/'+df['full_path'][i][2:-2])  
    label = df['gender'][i]
    if(image is not None):
        if(label==0 or label ==1):
#            if(df['gender'][i] == 1 ):
#                label = 'male'
#            else:
#                label ='female'
            pixels =     (image)
            rawImages.append(pixels)
            labels.append(label)


rawImages = np.array(rawImages)
print(rawImages[1])
labels = np.array(labels)
#print("[INFO] pixels matrix: {:.2f}MB".format(
#	rawImages.nbytes / (1024 * 1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.20, random_state=42)


test_datagen = ImageDataGenerator(rescale=1./255)

history= classifier.fit_generator(train_datagen.flow(trainRI, trainRL, batch_size=32),
                      steps_per_epoch=625,
                        epochs = 200,verbose=1,validation_data=test_datagen.flow(testRI, testRL),validation_steps = 137)
print(classifier.fit_generator(train_datagen.flow(trainRI, trainRL, batch_size=32),
                      steps_per_epoch=625,
                        epochs = 200,verbose=1,validation_data=test_datagen.flow(testRI, testRL),validation_steps = 137))
from tensorflow.keras.preprocessing import image
test_image = image.load_img('F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/77/315077_1966-04-20_2007.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
if result[0][0] == 1:
    prediction = 'male'
else:
    prediction = 'female'