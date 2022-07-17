# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:13:11 2022

@author: Narendra_IITJ
"""
""" 
1 : car    : 2754 objects
2 : truck  : 614 objects
9 : van    : 202 objets
11: pickup : 1910 objets
"""
import pandas as pd
import os
import numpy as np
import cv2
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from imutils import paths
import matplotlib.pyplot as plt
import argparse


#%%
start = time.time()
Annotations = []
directory = 'Annotations512'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    df = pd.read_csv(f,header = None,sep = ' ')
    locations = []
    for i in range(len(df)):
        n1 = int(df.iloc[i,0])
        n2 = int(df.iloc[i,1])
        n11 = max(n1-10,0)
        n12 = min(n1+10,511)
        n21 = max(n2-10,0)
        n22 = min(n2+10,511)
        label = int(df.iloc[i,3])
        locations.append([[n11, n12, n21, n22],[label]])
    
    Annotations.append(locations)
    Annotations.append(locations)
    
end = time.time()
print("process took",end-start,"seconds")
    
#%%
start = time.time()
directory = 'Vehicules512'
INPUT_SIZE = (128, 128)
index = 0
data = []
labels = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    orig = cv2.imread(f)
    for annotation in Annotations[index]:
        label = annotation[1][0]
        if(label == 1 or label == 11):
            roiOrig = orig[annotation[0][2]:annotation[0][3], annotation[0][0]:annotation[0][1]]
            roi = cv2.cvtColor(roiOrig, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, INPUT_SIZE)

            data.append(roi)
            labels.append(label)
    index += 1
        
end = time.time()
print("process took",end-start,"seconds")

#%%

car = 0
truck = 0
van = 0
pickup = 0
for lable in labels:
  
    if(lable == 1): 
        car+=1
    if(lable == 2): 
        truck +=1
    if(lable == 9): 
        van += 1
    if(lable == 11): 
        pickup += 1

    
print("car = ",car)
print("truck = ",truck)
print("van = ",van)
print("pickup = ",pickup)


#%%

ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=10,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#%%
start = time.time()

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

end = time.time()
print("process took",end-start,"seconds")


#%%
start = time.time()

baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(128, 128, 3)))


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False
    
opt = Adam(lr=1e-4)

model.compile(loss= "binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

end = time.time()
print("process took",end-start,"seconds")
#%%
start = time.time()

H = model.fit(
	x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

model.save('VGG16_2class.h5')
model.save_weights('VGG16_2class_weights')

end = time.time()
print("process took",end-start,"seconds")
#%%
start = time.time()
predictions = model.predict(x=testX.astype("float32"), batch_size=32)

end = time.time()
print("process took",end-start,"seconds")
#%%
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),target_names=['car','pickup']))

print()
from sklearn.metrics import confusion_matrix, accuracy_score
predict = predictions.argmax(axis = 1)
y_true = testY.argmax(axis = 1)

print(confusion_matrix(y_true, predict))

print()
print(accuracy_score(y_true, predict))

#%%
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(args["plot"])


N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["plot"])




