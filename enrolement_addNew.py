
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

            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            image= cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

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
                    x1,y1,w1,h1= convert_and_trim_bb(gray, rect1) 
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
        
        face_cascade = cv2.CascadeClassifier(FACE_DETECTION_MODELS+'haarcascade_frontalface_default.xml')
        ret, frame = self.capture.read()
        directory = self.class_name_input.text()
        path = GALLERY_IMAGES_PATH+ str(directory) + "/"
        """
                if os.path.isdir(path):
            print("Path Already Exists...")
            print("Removing " + str(directory))
            shutil.rmtree(GALLERY_IMAGES_PATH+"{}".format(str(directory)), ignore_errors=True)
            print(str(directory) + 'Has been removed')
            print("Creating New folder named : " + str(directory))
            os.mkdir(path)
        else:
            os.mkdir(path)
        """
        face_frames1_labels = [self.face_frame1_1, self.face_frame1_2, self.face_frame1_3, self.face_frame1_4, self.face_frame1_5]
        face_frames2_labels = [self.face_frame2_1, self.face_frame2_2, self.face_frame2_3, self.face_frame2_4, self.face_frame2_5]
        frame_number = 5
        #self.class_name_input.setEnabled(False)
        t_seconds=15
        count = 0

        while ret:
            start_time = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            print(faces)
            print(type(faces))
            print('===========================')
            try:
                if type(faces)==tuple:
                    break
                else: 
                    for (x, y, w, h) in faces:
                        
                        roi_color = frame[y:y + h, x:x + w]
                        if count ==0:
                            roi_color= cv2.resize(roi_color, (170,170), interpolation=cv2.INTER_CUBIC)
                        else :
                            roi_color= cv2.resize(roi_color, (85,85), interpolation=cv2.INTER_CUBIC)
                        height, width, channel = roi_color.shape
                        print(roi_color.shape)
                        step = channel * width
                        qImg = QImage(roi_color.data, width, height, step, QImage.Format_RGB888)

                        face_frames2_labels[count].setPixmap(QPixmap.fromImage(qImg))
                        #captured_img = cv2.imwrite(path + "frame-{}.jpg".format(count)  , roi_color)
                        ret, frame = self.capture.read()
                        time.sleep(2.0 - time.time() + start_time)
                        count += 1
                        '''
                        if count == 25 or count == 40:
                            self.notif.setText("Please change position")
                            self.notif.setStyleSheet("""color : white;""")

                        print("Picture " + str(count) + " Saved successfully")
                        '''
                    
            except:
                pass
            if count == frame_number:
                break
        self.viewCam()



    def feature_extraction(self):
        model_type = "vgg16"

        name = self.class_name_input.text()
        image_list = []
        feature_vectors = []
        labels = []
        images = glob(GALLERY_IMAGES_PATH + str(name) + "/"+ "*.jpg")
        file_x_train = GALLERY_PATH+'vgg16_x_train.npy'
        file_y_train = GALLERY_PATH+'vgg16_y_train.npy'
        x_train = np.load(file_x_train)
        y_train = np.load(file_y_train)
        authorized = np.load(HISTORY_PATH+"authorized.npy")

        vgg_face_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        for image in images:
            roi_color = cv2.imread(image)
            roi_color = cv2.resize(roi_color, (224, 224),interpolation=cv2.INTER_AREA)  # load an image and resize it to 224,224 like vgg face input size
            roi_color = img_to_array(roi_color)  # convert the image to an array
            roi_color = np.expand_dims(roi_color,  axis=0)
            roi_color = utils.preprocess_input(roi_color, version= 1)  # preprocess the image 1 = vggface resnet = 2)
            feature_vector = vgg_face_model.predict(roi_color)  # extract the features
            np.shape(feature_vector)
            feature_vectors.append(feature_vector)#append the current feature vector to the gallery
            labels.append(str(name))#append the current label to the gallery


        feature_vectors = np.squeeze(np.array(feature_vectors), axis=1)
        labels = np.expand_dims(np.array(labels), axis=1)
        #exception si fichier n'existe pas np.save else concatenate
        feature_vectors = np.concatenate((x_train, feature_vectors), axis=0)
        labels = np.concatenate((y_train, labels), axis=0)
        data_filename = GALLERY_PATH + str(model_type) + '_x_train.npy'      
        labels_filename = GALLERY_PATH + str(model_type) + '_y_train.npy' 
        np.save(data_filename, feature_vectors)
        np.save(labels_filename, labels)

        authorized = np.append(authorized, name)

        print('Saved to : {}/{}'.format(GALLERY_PATH, data_filename))
        print('Saved to : {}/{}'.format(GALLERY_PATH, labels_filename))
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
