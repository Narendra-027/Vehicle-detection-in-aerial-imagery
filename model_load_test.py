# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:23:05 2022

@author: Narendra_IITJ
"""

from keras.models import load_model
import pandas as pd
import os
import numpy as np
import cv2
import time

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

data = np.array(data)
labels = np.array(labels)

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#%%

model = load_model(r'path of the model location on your machine')

start = time.time()
predictions = model.predict(x=data, batch_size=32)
predict = predictions.argmax(axis = 1)

end = time.time()
print("process took",end-start,"seconds")

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_true = labels.argmax(axis = 1)

print(classification_report(labels.argmax(axis=1),
	predictions.argmax(axis=1),target_names=['car','pickup']))

print()

print("counfusion matrix")
print(confusion_matrix(y_true, predict))

print()
print("accuracy: ",accuracy_score(y_true, predict))
