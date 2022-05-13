
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
from utils import prediction_cosine_similarity2, findCosineDistance
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

        #font21
        font23 = QtGui.QFont("Play", 23)
        font21 = QtGui.QFont("Play", 21)
        font11 = QtGui.QFont("Play", 11)
        font7 = QtGui.QFont("Play", 7)
        font4 = QtGui.QFont("Play", 4)

        #Main Window
        enrolement_addNew.setObjectName("enrolement_addNew")
        enrolement_addNew.resize(900, 500)
        enrolement_addNew.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        enrolement_addNew.setWindowIcon(QIcon("./imgs/logo.jpg"))

        #Central Widget
        self.centralwidget = QtWidgets.QWidget(enrolement_addNew)
        self.centralwidget.setObjectName("centralwidget")

        #Frame Capture
        self.frame = QtWidgets.QLabel(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(420, 15, 451, 441))
        self.frame.setFont(font21)
        self.frame.setObjectName("frame")
        self.frame.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                                     )
        self.frame.setAlignment(QtCore.Qt.AlignCenter)
        
        #Taken Picture Areas

        #Frame 1 (Main Frame)
        self.face_frame1 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame1.setGeometry(QtCore.QRect(15, 140, 170, 170 ))
        self.face_frame1.setObjectName("face_frame1")
        self.face_frame1.setScaledContents(True)
        self.face_frame1.setFont(font4)
        self.face_frame1.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame1.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 2 
        self.face_frame2 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame2.setGeometry(QtCore.QRect(15, 371, 85, 85 ))
        self.face_frame2.setObjectName("face_frame2")
        self.face_frame2.setScaledContents(True)
        self.face_frame2.setFont(font4)
        self.face_frame2.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame2.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 3 
        self.face_frame3 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame3.setGeometry(QtCore.QRect(115, 371, 85, 85 ))
        self.face_frame3.setObjectName("face_frame3")
        self.face_frame3.setScaledContents(True)
        self.face_frame3.setFont(font4)
        self.face_frame3.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame3.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 4
        self.face_frame4 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame4.setGeometry(QtCore.QRect(215, 371, 85, 85 ))
        self.face_frame4.setObjectName("face_frame4")
        self.face_frame4.setScaledContents(True)
        self.face_frame4.setFont(font4)
        self.face_frame4.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame4.setAlignment(QtCore.Qt.AlignCenter)
        #Frame 5 
        self.face_frame5 = QtWidgets.QLabel(self.centralwidget)
        self.face_frame5.setGeometry(QtCore.QRect(315, 371, 85, 85 ))
        self.face_frame5.setObjectName("face_frame5")
        self.face_frame5.setScaledContents(True)
        self.face_frame5.setFont(font4)
        self.face_frame5.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame5.setAlignment(QtCore.Qt.AlignCenter)


        #Start Recording Button
        self.startBTN = QtWidgets.QPushButton(self.centralwidget)
        self.startBTN.setGeometry(QtCore.QRect(45, 45, 142, 40))
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
        self.take_pic.setGeometry(QtCore.QRect(200, 45, 148, 40))
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
        
        self.take_pic.hide()
        self.take_pic.clicked.connect(self.capture_image)

        #NameText (Class Name Input)
        self.class_name_input = QtWidgets.QLineEdit(self.centralwidget)
        self.class_name_input.setGeometry(QtCore.QRect(205, 235, 200, 30))
        self.class_name_input.setEnabled(False)
        self.class_name_input.setPlaceholderText('Enroll As (Last Name)')
        self.class_name_input.setStyleSheet("border: 1px solid white;"
                      "font-style:bold;"
                      "background-color:rgb(255,255,255);"
                                    "color: rgb(0,0,0);"
                      )


        self.enrolementBTN = QtWidgets.QPushButton(self.centralwidget)
        self.enrolementBTN.setGeometry(QtCore.QRect(220, 275, 171, 35))
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

        self.capture = cv2.VideoCapture(0)
        
        self.enrolementBTN.clicked.connect(self.feature_extraction)
        

    def viewCam(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_DETECTION_MODELS+'haarcascade_frontalface_default.xml')
        while True:
            ret, frame = self.capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = self.face_cascade.detectMultiScale(gray)
            height, width, channel = image.shape
            step = channel * width

            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.frame.setPixmap(QPixmap.fromImage(qImg))
            self.take_pic.show()

            if cv2.waitKey(0) & 0xFF == ord('z'):
                break

        self.capture.release()
        cv2.destroyAllWindows()




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

        face_frames_labels = [self.face_frame1, self.face_frame2, self.face_frame3, self.face_frame4, self.face_frame5]
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

                        face_frames_labels[count].setPixmap(QPixmap.fromImage(qImg))
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
        self.frame.setText(_translate("enrolement_addNew", "Camera Capture Frame"))
        self.face_frame1.setText(_translate("MainWindow", "Image \n 1"))
        self.face_frame2.setText(_translate("MainWindow", "Image \n 2"))
        self.face_frame3.setText(_translate("MainWindow", "Image \n 3"))
        self.face_frame4.setText(_translate("MainWindow", "Image \n 4"))
        self.face_frame5.setText(_translate("MainWindow", "Image \n 5"))
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
