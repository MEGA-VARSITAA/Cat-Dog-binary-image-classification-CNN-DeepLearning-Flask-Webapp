# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:52:41 2023

@author: megav
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import warnings
import os
#import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from keras.utils import load_img

from keras import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

#form keras_preprocessing import load_img
warnings.filterwarnings('ignore')
input_path=[]
label=[]

for class_name in os.listdir("d:/ML Project/Datasets/PetImages"):
    for path in os.listdir("d:/ML Project/Datasets/PetImages/"+class_name):
        if class_name=='Cat':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join( "D:/ML Project/Datasets/PetImages",class_name,path))
        
#print(input_path[0],label[0])
input_path=np.array(input_path)
label=np.array(label)
 
df=pd.DataFrame()
df['images']=input_path
df['label']=label
df=df.sample(frac=1).reset_index(drop=True)
#print(df.head())

#print (l)

#df=df[df['images']!='D:/ML Project/Datasets/PetImages/Cat/0.jpg 0']
df=df[df['images']!='D:/ML Project/Datasets/PetImages/Dog/Thumbs.db']
df=df[df['images']!='D:/ML Project/Datasets/PetImages/Cat/Thumbs.db']
#df=df[df['images']!='C:/Users/megav/OneDrive/Desktop/ML Project/Datasets/PetImages/Dog/11702.jpg']
#df=df[df['images']!='C:/Users/megav/OneDrive/Desktop/ML Project/Datasets/PetImages/Cat/0.jpg']
print(len(df))


df['label']=df['label'].astype('str')

from sklearn.model_selection import train_test_split 
train,test=train_test_split(df,test_size=0.2,random_state=42)

model=Sequential([
Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
MaxPool2D((2,2)),
Conv2D(32,(3,3),activation='relu'),
MaxPool2D(2,2),
Conv2D(64,(3,3),activation='relu'),
MaxPool2D(2,2),
Flatten(),
Dense(512,activation='relu'),
Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 

kf=KFold(n_splits= 5,shuffle=True, random_state=1)
skf= StratifiedKFold(n_splits=5,random_state=7,shuffle=True)


from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

val_generator=ImageDataGenerator(rescale=1./255)

scores=list()
FoldSetno = 0
for train_index, test_index in kf.split(input_path):
    x_train,y_train,x_test,y_test=input_path[train_index],label[train_index],input_path[test_index],label[test_index]
    
    train_iterator=train_generator.flow_from_dataframe(
        train,
        x_col='images',
        y_col='label',
        target_size=(128,128),
        batch_size=512,
        class_mode='binary'
        )
    val_iterator=train_generator.flow_from_dataframe(
        test,
        x_col='images',
        y_col='label',
        target_size=(128,128),
        batch_size=512,
        class_mode='binary'
        )
  
    
    history=model.fit(train_iterator, epochs=25 ,validation_data=val_iterator) 
    scores.append({'acc':np.average(history.history['accuracy']),'val_acc':np.average(history.history['val_accuracy'])})
    FoldSetno+=1
    
from matplotlib import pyplot as plt

train_d=[]
validation_d=[]
plt.subplot(1,1,1)
for s in scores:
    train_d.append(s['acc'])
    validation_d.append(s['val_acc'])
print(train_d)
print(validation_d)

plt.plot(train_d, color='blue',label='train')
plt.plot(validation_d,color='red',label='validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

  




image_path="D:/ML Project/Datasets/ash-v0_MCllHY9M-unsplash.jpg"
img=load_img(image_path,target_size=(128,128))
img=np.array(img)
img=img/255.0
img=img.reshape(1,128,128,3)
pred=model.predict(img)
if(pred>0.5):
    label='Dog'
else:
    label='Cat'
print (label)

filename='model.h5'
model.save(filename)

from tensorflow.keras.models import load_model
loaded_model= load_model(filename)
pred1=loaded_model.predict(img)
