import streamlit as st 
import numpy as np
import cv2
import pandas as pd

#********************

def main():
    st.title('Recognition App')
    st.sidebar.title('Parameters')
    st.sidebar.markdown('Choose Parameters')
    st.subheader('Choose What You want to Identify')
    a = st.selectbox("Choose Option", ("Face", "Smile", "Eyes"))

    if a=='Face':
        b=st.sidebar.button('Capture Images For Face Recognition 🧔',key='capface')
        if b:
            face_model=cv2.CascadeClassifier('C:/Users/kevin/Documents/test/HaarCascadeFace.xml')
            cap=cv2.VideoCapture(0)
            while True:
                status,photo=cap.read()
                face_cor=face_model.detectMultiScale(photo)
                if len(face_cor)==0:
                    pass
                else:
                    x1=face_cor[0][0]
                    y1=face_cor[0][1]
                    x2=x1+face_cor[0][2]
                    y2=y1+face_cor[0][3]
                    photo=cv2.rectangle(photo,(x1,y1),(x2,y2),[0,255,0])
                    cv2.putText(photo, "Hey Kevin", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Look At You',photo)
                    if cv2.waitKey(10)==13:
                        break          
            cv2.destroyAllWindows()
            cap.release()
        st.sidebar.markdown('To Stop, Press Enter')


    if a=='Smile':
        c=st.sidebar.button('Capture Images For Smile Recognition 😁',key='capnose')
        if c:
            smile_model=cv2.CascadeClassifier('C:/Users/kevin/Documents/test/HaarCascadeSmile.xml')
            cap=cv2.VideoCapture(0)
            while True:
                status,photos=cap.read()
                smile_cor=smile_model.detectMultiScale(photos)
                if len(smile_cor)==0:
                    pass
                else:
                    x1=smile_cor[0][0]
                    y1=smile_cor[0][1]
                    x2=x1+smile_cor[0][2]
                    y2=y1+smile_cor[0][3]
                    photos=cv2.rectangle(photos,(x1,y1),(x2,y2),[0,255,0])
                    cv2.putText(photos, "Hey Kevin's Smile", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Look At Your Smile',photos)
                    if cv2.waitKey(10)==13:
                        break          
            cv2.destroyAllWindows()
            cap.release()
        st.sidebar.markdown('To Stop, Press Enter')
    
    if a=='Eyes':
        d=st.sidebar.button('Capture Images For Eyes Recognition 👀',key='capeyes')
        if d:
            eyes_model=cv2.CascadeClassifier('C:/Users/kevin/Documents/test/HaarCascadeEyes.xml')
            cap=cv2.VideoCapture(0)
            while True:
                status,photoe=cap.read()
                eyes_cor=eyes_model.detectMultiScale(photoe)
                if len(eyes_cor)==0:
                    pass
                else:
                    x1=eyes_cor[0][0]
                    y1=eyes_cor[0][1]
                    x2=x1+eyes_cor[0][2]
                    y2=y1+eyes_cor[0][3]
                    photoe=cv2.rectangle(photoe,(x1,y1),(x2,y2),[0,255,0])
                    cv2.putText(photoe, "Hey Kevin's Eyes", (250, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Look At Your Eyes',photoe)
                    if cv2.waitKey(10)==13:
                        break          
            cv2.destroyAllWindows()
            cap.release()
        st.sidebar.markdown('To Stop, Press Enter')

if __name__ == '__main__':
    main()