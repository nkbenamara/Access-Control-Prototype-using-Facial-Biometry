
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QCursor, QIcon, QFont
from PyQt5.QtWidgets import QToolTip,QProgressBar,QVBoxLayout
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QLineEdit
from tqdm import tqdm
import sys
from datetime import datetime
import os
import cv2
import pandas
import random
import string
import os
from glob import glob
import numpy as np
import asyncio

#import pycuda.autoinit
import time
import queue
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from utils import prediction_cosine_similarity2, findCosineDistance, convert_and_trim_bb
import dlib

from models import *
from paths import *

class Ui_enrolement_addNew(object):
    def setupUi(self, enrolement_addNew):
        '''
        MAIN WINDOW - ENROLEMENT_ADDNEW
        '''
        #Define Fonts
        QtGui.QFontDatabase.addApplicationFont("./fonts/Play-Regular.ttf")

        #font21/7/4
        font21 = QtGui.QFont("Play", 21)
        font7 = QtGui.QFont("Play", 7)
        font4 = QtGui.QFont("Play", 4)
        font15 = QtGui.QFont("Play",15)

        #Main Window
        enrolement_addNew.setObjectName("enrolement_addNew")
        enrolement_addNew.resize(860, 830)
        enrolement_addNew.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        enrolement_addNew.setWindowIcon(QIcon("./imgs/logo.jpg"))

        #Central Widget
        self.centralwidget = QtWidgets.QWidget(enrolement_addNew)
        self.centralwidget.setObjectName("centralwidget")
        
        #Frame Capture 1
        self.cam_label1 = QtWidgets.QLabel(self.centralwidget)
        self.cam_label1.setGeometry(QtCore.QRect(15, 15, 400, 400))
        self.cam_label1.setFont(font21)
        self.cam_label1.setObjectName("cam_label1")
        self.cam_label1.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                                     )
        self.cam_label1.setAlignment(QtCore.Qt.AlignCenter)

        #Frame Capture 2
        self.cam_label2 = QtWidgets.QLabel(self.centralwidget)
        self.cam_label2.setGeometry(QtCore.QRect(450, 15, 400, 400))
        self.cam_label2.setFont(font21)
        self.cam_label2.setObjectName("cam_label2")
        self.cam_label2.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                                     )
        self.cam_label2.setAlignment(QtCore.Qt.AlignCenter)
        
        #NoFace Detected Area 1 
        self.No_Face_1 = QtWidgets.QLabel(self.centralwidget)
        self.No_Face_1.setGeometry(QtCore.QRect(140, 350, 150, 30))
        self.No_Face_1.setObjectName("No_Face_1")
        self.No_Face_1.setFont(font15)
        self.No_Face_1.setStyleSheet("background-color:rgba(255,75,60,200);"
                                    "color: white;"
                                     "font-style:bold;"
                                     )
        self.No_Face_1.setAlignment(QtCore.Qt.AlignCenter)
        self.No_Face_1.setVisible(False)
        
        #NoFace Detected Area 2
        self.No_Face_2 = QtWidgets.QLabel(self.centralwidget)
        self.No_Face_2.setGeometry(QtCore.QRect(575, 350, 150, 30))
        self.No_Face_2.setObjectName("No_Face_2")
        self.No_Face_2.setFont(font15)
        self.No_Face_2.setStyleSheet("background-color:rgba(255,75,60,200);"
                                    "color: white;"
                                     "font-style:bold;"
                                     )
        self.No_Face_2.setAlignment(QtCore.Qt.AlignCenter)
        self.No_Face_2.setVisible(False)
        #Taken Picture Areas 
        #Frame 1_1 (Main Frame 1)
        self.face_frame1_1 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1_1.setGeometry(QtCore.QRect(230, 500, 185, 185 ))
        self.face_frame1_1.setObjectName("face_frame1_1")
        self.face_frame1_1.setScaledContents(True)
        self.face_frame1_1.setFont(font4)
        self.face_frame1_1.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1_1.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 1_2 
        self.face_frame1_2 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1_2.setGeometry(QtCore.QRect(130, 500, 85, 85 ))
        self.face_frame1_2.setObjectName("face_frame1_2")
        self.face_frame1_2.setScaledContents(True)
        self.face_frame1_2.setFont(font4)
        self.face_frame1_2.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1_2.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 1_3 
        self.face_frame1_3 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1_3.setGeometry(QtCore.QRect(30, 500, 85, 85 ))
        self.face_frame1_3.setObjectName("face_frame1_3")
        self.face_frame1_3.setScaledContents(True)
        self.face_frame1_3.setFont(font4)
        self.face_frame1_3.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1_3.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 1_4
        self.face_frame1_4 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1_4.setGeometry(QtCore.QRect(130, 600, 85, 85 ))
        self.face_frame1_4.setObjectName("face_frame1_4")
        self.face_frame1_4.setScaledContents(True)
        self.face_frame1_4.setFont(font4)
        self.face_frame1_4.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1_4.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 1_5 
        self.face_frame1_5 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1_5.setGeometry(QtCore.QRect(30, 600, 85, 85 ))
        self.face_frame1_5.setObjectName("face_frame1_5")
        self.face_frame1_5.setScaledContents(True)
        self.face_frame1_5.setFont(font4)
        self.face_frame1_5.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1_5.setAlignment(QtCore.Qt.AlignCenter)

        #Frame 2_1 (Main Frame 2)
        self.face_frame2_1 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2_1.setGeometry(QtCore.QRect(450, 500, 185, 185 ))
        self.face_frame2_1.setObjectName("face_frame2_1")
        self.face_frame2_1.setScaledContents(True)
        self.face_frame2_1.setFont(font4)
        self.face_frame2_1.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2_1.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 2_2 
        self.face_frame2_2 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2_2.setGeometry(QtCore.QRect(650, 500, 85, 85 ))
        self.face_frame2_2.setObjectName("face_frame2_2")
        self.face_frame2_2.setScaledContents(True)
        self.face_frame2_2.setFont(font4)
        self.face_frame2_2.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2_2.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 2_3 
        self.face_frame2_3 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2_3.setGeometry(QtCore.QRect(750, 500, 85, 85 ))
        self.face_frame2_3.setObjectName("face_frame2_3")
        self.face_frame2_3.setScaledContents(True)
        self.face_frame2_3.setFont(font4)
        self.face_frame2_3.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2_3.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 2_4
        self.face_frame2_4 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2_4.setGeometry(QtCore.QRect(650, 600, 85, 85 ))
        self.face_frame2_4.setObjectName("face_frame2_4")
        self.face_frame2_4.setScaledContents(True)
        self.face_frame2_4.setFont(font4)
        self.face_frame2_4.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2_4.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 2_5 
        self.face_frame2_5 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2_5.setGeometry(QtCore.QRect(750, 600, 85, 85 ))
        self.face_frame2_5.setObjectName("face_frame2_5")
        self.face_frame2_5.setScaledContents(True)
        self.face_frame2_5.setFont(font4)
        self.face_frame2_5.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2_5.setAlignment(QtCore.Qt.AlignCenter)


        #Start Recording Button
        self.startBTN = QtWidgets.QPushButton(self.centralwidget)
        self.startBTN.setGeometry(QtCore.QRect(285, 450, 142, 40))
        self.startBTN.setFont(font7)
        self.startBTN.setStyleSheet(
            "QPushButton::hover"
                             "{"
                             "background-color:rgb(255,255,255);"
                            "color: black;"
                             "}"
                            "border: 1px solid white;"
        )
        self.startBTN.setIcon(QIcon("./imgs/start.ico"))
        self.startBTN.setIconSize(QtCore.QSize(20, 20))
        self.startBTN.setShortcut('Ctrl+R')
        self.startBTN.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.startBTN.setObjectName("startBTN")
        self.startBTN.clicked.connect(self.viewCam)

        #Capture Button
        self.take_pic = QtWidgets.QPushButton(self.centralwidget)
        self.take_pic.setGeometry(QtCore.QRect(430, 450, 142, 40))
        self.take_pic.setFont(font7)
        self.take_pic.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.take_pic.setIcon(QIcon("./imgs/cam.png"))
        self.take_pic.setIconSize(QtCore.QSize(30, 30))
        self.take_pic.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.take_pic.setObjectName("take_pic")
        
        self.take_pic.setEnabled(False)
        self.take_pic.clicked.connect(self.capture_image)
        
        #NameText (Class Name Input)
        self.class_name_input = QtWidgets.QLineEdit(self.centralwidget)
        self.class_name_input.setGeometry(QtCore.QRect(330, 700, 200, 30))
        self.class_name_input.setEnabled(False)
        self.class_name_input.setPlaceholderText('Enroll As (Last Name)')
        self.class_name_input.setStyleSheet("border: 1px solid white;"
                      "font-style:bold;"
                      "background-color:rgb(255,255,255);"
                                    "color: rgb(0,0,0);"
                      )


        self.enrolementBTN = QtWidgets.QPushButton(self.centralwidget)
        self.enrolementBTN.setGeometry(QtCore.QRect(345, 745, 170, 40))
        self.enrolementBTN.setEnabled(False)
        self.enrolementBTN.setObjectName("Enrolement Boutton")
        self.enrolementBTN.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.enrolementBTN.setIcon(QIcon("./imgs/scan.png"))
        self.enrolementBTN.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.enrolementBTN.setIconSize(QtCore.QSize(20, 20))
        self.enrolementBTN.clicked.connect(self.feature_extraction)

        font = QtGui.QFont()
        font.setPointSize(12)

        self.enrolementBTN.setFont(font)
        self.enrolementBTN.setToolTip("Starts feature extraction")

        self.notif = QtWidgets.QLabel(self.centralwidget)
        self.notif.setGeometry(QtCore.QRect(600, 18, 265, 21))
        self.notif.setObjectName("notif")

        font = QtGui.QFont()
        font.setPointSize(15)

        self.notif.setFont(font)


        enrolement_addNew.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(enrolement_addNew)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 21))
        self.menubar.setObjectName("menubar")

        enrolement_addNew.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(enrolement_addNew)
        self.statusbar.setObjectName("statusbar")

        enrolement_addNew.setStatusBar(self.statusbar)

        self.retranslateUi(enrolement_addNew)
        QtCore.QMetaObject.connectSlotsByName(enrolement_addNew)

        self.capture = cv2.VideoCapture(1)
        
        self.enrolementBTN.clicked.connect(self.feature_extraction)

        self.camera_labels=[self.cam_label1, self.cam_label2] 

    def viewCam(self):
        self.capture1 = cv2.VideoCapture(1)
        self.capture2 = cv2.VideoCapture(0)
        self.cameras=[self.capture1, self.capture2]

        while hasattr(self.capture1, 'read'):

            ret1, frame1 = self.cameras[0].read()
            ret2, frame2 = self.cameras[1].read()


            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            rects1 = detector(gray1, 0)
            rects2 = detector(gray2, 0)
                
            image1= cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image2= cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
     
            height1, width1, channel1 = image1.shape
            height2, width2, channel2 = image2.shape

            step1 = channel1 * width1
            step2 = channel2 * width2

            qImg1 = QImage(image1.data, width1, height1, step1, QImage.Format_RGB888)
            qImg2 = QImage(image2.data, width2, height2, step2, QImage.Format_RGB888)
            
            # Draw a rectangle around the faces (Camera 1)
            if len(rects1)!=0:
                self.No_Face_1.setVisible(True)
                self.No_Face_1.setStyleSheet(
                                        "background-color:rgba(3,174,80,200);"
                                        "font-style:bold;"
                                        "color: white;"
                                        )
                self.No_Face_1.setText("Face Detected")
                
                for rect1 in rects1: 
                    x1,y1,w1,h1= convert_and_trim_bb(gray1, rect1) 
                    cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            else:
                self.No_Face_1.setStyleSheet(
                                        "background-color:rgba(255,75,60,200);"
                                        "font-style:bold;"
                                        "color: white;"
                                        )
                self.No_Face_1.setText("No Face Detected")
                self.No_Face_1.setVisible(True)

            # Draw a rectangle around the faces (Camera 2)
            if len(rects2)!=0:
                self.No_Face_2.setVisible(True)
                self.No_Face_2.setStyleSheet(
                                        "background-color:rgba(3,174,80,200);"
                                        "font-style:bold;"
                                        "color: white;"
                                        )
                self.No_Face_2.setText("Face Detected")

                for rect2 in rects2:
                    x2,y2,w2,h2= convert_and_trim_bb(gray2, rect2) 
                    cv2.rectangle(image2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            else:
                self.No_Face_2.setStyleSheet(
                                        "background-color:rgba(255,75,60,200);"
                                        "font-style:bold;"
                                        "color: white;"
                                        )
                self.No_Face_2.setText("No Face Detected")
                self.No_Face_2.setVisible(True)

            if len(rects1)!=0 and len(rects2)!=0:
                self.take_pic.setEnabled(True)
            else:
                self.take_pic.setEnabled(False) 

            self.camera_labels[0].setPixmap(QPixmap.fromImage(qImg1))
            self.camera_labels[1].setPixmap(QPixmap.fromImage(qImg2))
            
                            # Draw a rectangle around the faces (Camera 1)


            if cv2.waitKey(0) & 0xFF == ord('z'):
                break




    def capture_image(self):
        
        face_frames1_labels = [self.face_frame1_1, self.face_frame1_2, self.face_frame1_3, self.face_frame1_4, self.face_frame1_5]
        face_frames2_labels = [self.face_frame2_1, self.face_frame2_2, self.face_frame2_3, self.face_frame2_4, self.face_frame2_5]
        self.images_camera1=[]
        self.images_camera2=[]
        count = 0

        while True and count <5:

            ret1, frame1 = self.cameras[0].read()
            ret2, frame2 = self.cameras[1].read()

            print("count : {}".format(count))
            print("=====================")
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            rects1 = detector(gray1, 0)
            rects2 = detector(gray2, 0)
            for rect1 in rects1:
                x1,y1,w1,h1= convert_and_trim_bb(gray1, rect1)
                roi_color1 = frame1[y1:y1 + h1, x1:x1 + w1]
                
            for rect2 in rects2:
                x2,y2,w2,h2= convert_and_trim_bb(gray2, rect2)
                roi_color2 = frame2[y2:y2 + h2, x2:x2 + w2]
                    
            
            img1=cv2.cvtColor(roi_color1, cv2.COLOR_BGR2RGB)
            self.images_camera1.append(cv2.resize(img1, (224, 224),interpolation=cv2.INTER_AREA))
            qImg1 = QImage(img1, w1, h1, 3*w1, QImage.Format_RGB888)
            face_frames1_labels[count].setPixmap(QPixmap.fromImage(qImg1))
            
            
            img2=cv2.cvtColor(roi_color2, cv2.COLOR_BGR2RGB)
            self.images_camera2.append(cv2.resize(img2, (224, 224),interpolation=cv2.INTER_AREA))
            qImg2 = QImage(img2, w2, h2, 3*w2, QImage.Format_RGB888)
            face_frames2_labels[count].setPixmap(QPixmap.fromImage(qImg2))
                
            count+=1

            time.sleep(2)

        self.class_name_input.setEnabled(True)
        self.enrolementBTN.setEnabled(True)

        self.viewCam()



    def feature_extraction(self):

        name = self.class_name_input.text()
        os.makedirs(GALLERY_IMAGES_PATH +'Camera1/'+ str(name).capitalize(), exist_ok=True)
        os.makedirs(GALLERY_IMAGES_PATH + 'Camera2/'+ str(name).capitalize(), exist_ok=True)

        
        feature_vectors_camera1 = []
        feature_vectors_camera2 = []
        labels_camera1 = []
        labels_camera2 = []
        

        file_x_train_camera1 = GALLERY_PATH+'x_train_camera1.npy'
        file_y_train_camera1 = GALLERY_PATH+'y_train_camera1.npy'

        file_x_train_camera2 = GALLERY_PATH+'x_train_camera2.npy'
        file_y_train_camera2 = GALLERY_PATH+'y_train_camera2.npy'

        x_train_camera1 = np.load(file_x_train_camera1)
        y_train_camera1 = np.load(file_y_train_camera1)

        x_train_camera2 = np.load(file_x_train_camera2)
        y_train_camera2 = np.load(file_y_train_camera2)

        authorized = np.load(HISTORY_PATH+"authorized.npy")

        vgg_face_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        count_camera1=1
        for image_camera1 in self.images_camera1:
            image_camera1=cv2.cvtColor(image_camera1, cv2.COLOR_BGR2RGB)
            roi_color = cv2.resize(image_camera1, (224, 224),interpolation=cv2.INTER_AREA)
            cv2.imwrite(GALLERY_IMAGES_PATH +'Camera1/'+ str(name).capitalize()+'/image'+str(count_camera1)+'.jpg', roi_color)
            count_camera1+=1  # load an image and resize it to 224,224 like vgg face input size
            roi_color = img_to_array(roi_color)  # convert the image to an array
            roi_color = np.expand_dims(roi_color,  axis=0)
            roi_color = utils.preprocess_input(roi_color, version= 1)  # preprocess the image 1 = vggface resnet = 2)
            feature_vector = vgg_face_model.predict(roi_color)  # extract the features
            
            feature_vectors_camera1.append(feature_vector)#append the current feature vector to the gallery
            labels_camera1.append(str(name).capitalize())#append the current label to the gallery


        feature_vectors_camera1 = np.squeeze(np.array(feature_vectors_camera1), axis=1)
        feature_vectors_camera1 = np.concatenate((x_train_camera1, feature_vectors_camera1), axis=0)
        labels_camera1 = np.expand_dims(np.array(labels_camera1), axis=1)
        labels_camera1 = np.concatenate((y_train_camera1, labels_camera1), axis=0)
 
        np.save(file_x_train_camera1, feature_vectors_camera1)
        np.save(file_y_train_camera1, labels_camera1)

        count_camera2=1
        for image_camera2 in self.images_camera2:
            image_camera2=cv2.cvtColor(image_camera2, cv2.COLOR_BGR2RGB)
            roi_color = cv2.resize(image_camera2, (224, 224),interpolation=cv2.INTER_AREA)
            cv2.imwrite(GALLERY_IMAGES_PATH +'Camera2/'+ str(name).capitalize()+'/image'+str(count_camera2)+'.jpg', roi_color)
            count_camera2+=1  # load an image and resize it to 224,224 like vgg face input size
            roi_color = img_to_array(roi_color)  # convert the image to an array
            roi_color = np.expand_dims(roi_color,  axis=0)
            roi_color = utils.preprocess_input(roi_color, version= 1)  # preprocess the image 1 = vggface resnet = 2)
            feature_vector = vgg_face_model.predict(roi_color)  # extract the features
            
            feature_vectors_camera2.append(feature_vector)#append the current feature vector to the gallery
            labels_camera2.append(str(name).capitalize())#append the current label to the gallery


        feature_vectors_camera2 = np.squeeze(np.array(feature_vectors_camera2), axis=1)
        feature_vectors_camera2 = np.concatenate((x_train_camera2, feature_vectors_camera2), axis=0)
        labels_camera2 = np.expand_dims(np.array(labels_camera2), axis=1)
        labels_camera2 = np.concatenate((y_train_camera2, labels_camera2), axis=0)
 
        np.save(file_x_train_camera2, feature_vectors_camera2)
        np.save(file_y_train_camera2, labels_camera2)



        authorized = np.append(authorized, str(name).capitalize())


        self.viewCam()
        #save_vector = np.concatenate((gallery, file_x_train), axis=1)



    def retranslateUi(self, enrolement_addNew):
        _translate = QtCore.QCoreApplication.translate
        enrolement_addNew.setWindowTitle(_translate("enrolement_addNew", "Add New Person"))
        self.cam_label1.setText(_translate("enrolement_addNew", "Camera Capture Frame 1"))
        self.cam_label2.setText(_translate("enrolement_addNew", "Camera Capture Frame 2"))
        self.face_frame1_1.setText(_translate("MainWindow", "Image \n 1 \n (Camera 1)"))
        self.face_frame1_2.setText(_translate("MainWindow", "Image \n 2"))
        self.face_frame1_3.setText(_translate("MainWindow", "Image \n 3"))
        self.face_frame1_4.setText(_translate("MainWindow", "Image \n 4"))
        self.face_frame1_5.setText(_translate("MainWindow", "Image \n 5"))
        self.face_frame2_1.setText(_translate("MainWindow", "Image \n 1 \n (Camera 2)"))
        self.face_frame2_2.setText(_translate("MainWindow", "Image \n 2"))
        self.face_frame2_3.setText(_translate("MainWindow", "Image \n 3"))
        self.face_frame2_4.setText(_translate("MainWindow", "Image \n 4"))
        self.face_frame2_5.setText(_translate("MainWindow", "Image \n 5"))
        self.take_pic.setText(_translate("enrolement_addNew", "Take Pictures"))
        self.startBTN.setText(_translate("enrolement_addNew", "Open Camera"))
        self.enrolementBTN.setText(_translate("enrolement_addNew", "Enroll"))
        self.notif.setText(_translate("enrolement_addNew", ""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    enrolement_addNew = QtWidgets.QMainWindow()
    ui = Ui_enrolement_addNew()
    ui.setupUi(enrolement_addNew)
    enrolement_addNew.show()
    sys.exit(app.exec_())
