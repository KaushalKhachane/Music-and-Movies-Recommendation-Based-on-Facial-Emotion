from __future__ import division, print_function
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    # i=0

    # GR_dict={0:(0,255,0),1:(0,0,255)}
    # model = tf.keras.models.load_model('model.h5')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # output=[]
    # cap = cv2.VideoCapture(0)
    # while (i<=60):
    #     ret, img = cap.read()
    #     faces = face_cascade.detectMultiScale(img,1.05,5)

    #     for x,y,w,h in faces:

    #         face_img = img[y:y+h,x:x+w] 

    #         resized = cv2.resize(face_img,(224,224))
    #         reshaped=resized.reshape(1, 224,224,3)/255
    #         predictions = model.predict(reshaped)

    #         max_index = np.argmax(predictions[0])

    #         emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
    #         predicted_emotion = emotions[max_index]
    #         output.append(predicted_emotion)
            
            
            
    #         cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
    #         cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
    #         cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    #     i = i+1

    #     cv2.imshow('LIVE', img)
    #     key = cv2.waitKey(1)
    #     if key == 27: 
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         break
    # print(output)
    # cap.release()
    # cv2.destroyAllWindows()
    # final_output1 = st.mode(output)
    # return render_template("buttons.html",final_output=final_output1)

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier =load_model('model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)


    i=0
    while (i<=60):
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        i=i+1
        
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(label)
    cap.release()
    cv2.destroyAllWindows()
    # final_output1 = st.mode(label)
    # print(final_output1)
    return render_template("buttons.html",final_output=label)
    


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/Surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/Angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/Sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/Disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/Happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/Fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/Neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/Surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/Angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/Sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/Disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/Happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/Fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/Neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    
if __name__ == "__main__":
    app.run(debug=True)