# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:02:06 2023

@author: User
"""

from flask import Flask,render_template,request
from keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.utils import load_img
import numpy as np
app=Flask(__name__)

dic={0:'Cat',1:'Dog'}

model= load_model('E:/Project/firemodelskfold.h5')

model.make_predict_function()



def predict_label(img_path):
   img=load_img(img_path,target_size=(128,128))
   img=np.array(img)
   img=img/255.0
   img=img.reshape(1,128,128,3)
   pred=np.argmax(model.predict(img), axis=-1)
   if (pred<0.5):
       label='Fire'
   else:
       label='Not fire'  
   return label
   

@app.route("/",methods=['GET','POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "About page"

@app.route("/submit",methods=['GET','POST'])
def get_output():
    if request.method=='POST':
        img=request.files["my_image"]
        img_path="static/" +img.filename
        img.save(img_path)
        
        p=predict_label(img_path)
        
    return render_template("index.html", prediction=p,img_path=img_path)

if __name__ =='__main__':
    app.run()