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
from tensorflow.keras.layers import MaxPooling2D
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
INPUT_SIZE = (64, 64)
index = 0
data = []
labels = []
car_pickup_annotations = [] #startX, startY, endX, endY
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    orig = cv2.imread(f)
    localAnnotations = []
    for annotation in Annotations[index]:
        label = annotation[1][0]
        if(label == 1 or label == 11):
            roiOrig = orig[annotation[0][2]:annotation[0][3], annotation[0][0]:annotation[0][1]]
            roi = cv2.cvtColor(roiOrig, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, INPUT_SIZE)
            localAnnotations.append(annotation[0])
            data.append(roi)
            if(index%2 == 0):
                labels.append([label,1]) #color image
            else:
                labels.append([label,0]) #grey image
    index += 1
    car_pickup_annotations.append(localAnnotations)
        
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

data = np.array(data)
labels = np.array(labels)
IsColor = labels[:,1]
labels = labels[:,0]

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.c_[labels,IsColor]
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25,  random_state=42) #stratify=labels,

trainIsColor = trainY[:,2]
trainY = trainY[:,:2]

testIsColor = testY[:,2]
testY = testY[:,:2]
#%%
plt.imshow(data[3014])
plt.show()
#%%

start = time.time()
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
OrigVGGModel = VGG16()
print("################### Origional VGG16 model summary #######################")
print()
print(OrigVGGModel.summary())
print()

baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(64, 64, 3)))

print("######################## Base Model summary ###################################")
print()
print(baseModel.summary())
print()


headModel = baseModel.layers[-3].output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
#headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
#headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#print(headModel.summary())
model = Model(inputs=baseModel.input, outputs=headModel)

print("######################## Main Model summary ###################################")
print()
print(model.summary())

for layer in baseModel.layers:
	layer.trainable = False
    
opt = Adam(lr=1e-5)

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



end = time.time()
print("process took",end-start,"seconds")

#%%

print("saving the model: ")
start = time.time()

# model.save('VGG16_2Class_acc_76.h5')
# model.save_weights('VGG16_2class_acc_76')

end = time.time()
print("process took",end-start,"seconds")

#%%
start = time.time()
predictions = model.predict(x=testX.astype("float32"), batch_size=32)

end = time.time()
print("process took",end-start,"seconds")
#%%
print("classification report")
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),target_names=['car','pickup']))

print()
from sklearn.metrics import confusion_matrix, accuracy_score
predict = predictions.argmax(axis = 1)
predict_color = predict[testIsColor == 1]
predict_grey = predict[testIsColor == 0]

y_true = testY.argmax(axis = 1)
y_true_color = y_true[testIsColor == 1]
y_true_grey = y_true[testIsColor == 0]

print("############## Combined ###############")
print("counfusion matrix")
print(confusion_matrix(y_true, predict))

print()
print("accuracy: ",accuracy_score(y_true, predict))
print()


print("############## Color ###############")
print("counfusion matrix")
print(confusion_matrix(y_true_color, predict_color))

print()
print("accuracy: ",accuracy_score(y_true_color, predict_color))
print()


print("############## grey ###############")
print("counfusion matrix")
print(confusion_matrix(y_true_grey, predict_grey))

print()
print("accuracy: ",accuracy_score(y_true_grey, predict_grey))
#%%
directory = 'Vehicules512'
filename = '00000044_co.png'
f = os.path.join(directory, filename)
img = cv2.imread(f)
clone = img.copy()
INPUT_SIZE = (64, 64)
DataImg = []
for annotation in car_pickup_annotations[76]:
    roiOrig = img[annotation[2]:annotation[3], annotation[0]:annotation[1]]
    roi = cv2.cvtColor(roiOrig, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (64,64))
    DataImg.append(roi)
    
DataImg = np.array(DataImg)
pred = model.predict(x = DataImg.astype("float32"),batch_size=32)
pred = pred.argmax(axis = 1)

    
#%%

#%%
i = 0
for annotation in car_pickup_annotations[76]:
    predLabel = 'car'
    roiOrig = img[annotation[2]:annotation[3], annotation[0]:annotation[1]]
    roi = cv2.cvtColor(roiOrig, cv2.COLOR_BGR2RGB)
    roi = np.array(cv2.resize(roi, (64,64)))
    print(roi.shape)
    plt.imshow(roi)
    plt.show()
    
    predI = pred[i]
    if(predI == 0):
        predLabel = 'car'
    else:
        predLabel = 'pickup'
        
    startX = annotation[0]
    startY = annotation[2]
    endX   = annotation[1]
    endY   = annotation[3]
    i = i +1
    
    cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)
    if(startY -10>10):
        y = startY - 10
    else:
        y = startY + 10
    cv2.putText(clone, predLabel, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    plt.imshow(clone)

    
        



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
#plt.savefig(args["plot"])

N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
#plt.savefig(args["plot"])

