
from tkinter.tix import MAIN
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil
import cv2
from PyQt5.QtCore import QSize, QRect
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QCursor, QIcon, QFont
from utils import load_actual_collaborators
import os
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

from functools import partial

from paths import *

class Ui_enrolement_ChangeAccess(object):
    def setupUi(self, enrolement_ChangeAccess):
        '''
        MAIN WINDOW - ENROLEMENT_CHANGEACCESS
        '''

        #Define Fonts
        QtGui.QFontDatabase.addApplicationFont("./fonts/Play-Regular.ttf")

        #font21/7/4
        font21 = QtGui.QFont("Play", 21)
        font7 = QtGui.QFont("Play", 7)
        font4 = QtGui.QFont("Play", 4)

        #Main Window
        enrolement_ChangeAccess.setObjectName("enrolement_ChangeAccess")
        enrolement_ChangeAccess.setFixedSize(950, 650)
        enrolement_ChangeAccess.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        enrolement_ChangeAccess.setWindowIcon(QIcon("./imgs/logo.jpg"))

        #Central Widget
        self.centralwidget = QtWidgets.QWidget(enrolement_ChangeAccess)
        self.centralwidget.setObjectName("centralwidget")
        
  
        authorizations = np.load(HISTORY_PATH+'authorized.npy')

        options = load_actual_collaborators()
        user_number=1
        #instantiate multiple label/buttons objects with a loop (TO SOLVE !!!!!!!!!!!!!!!)

        self.label_name_objects=[]
        self.authorization_label_name_objects=[]
        self.remove_button_name_objects=[]
        self.grant_button_name_objects=[]


        for option in options:
            
            if user_number%5==0:
                column = 4
                row = (user_number//5) - 1 
            else: 
                column = user_number%5 -1
                row = user_number//5

            #User Label
            self.label_name = QtWidgets.QLabel(enrolement_ChangeAccess)
            self.label_name.setGeometry(15 + column * 185,40 + row* 270,170,170)
            self.label_name.setStyleSheet("border: 1px solid white;")
            self.label_name.setFont(font7)
            self.label_name.setAlignment(QtCore.Qt.AlignCenter)
            self.label_name.setScaledContents(True)
            
            path=glob(GALLERY_IMAGES_PATH+ option +"/*")[0]
            face_img = cv2.imread(path)
            height, width, channel = face_img.shape
            step = channel * width
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            qImg = QImage(face_img.data, width, height, step, QImage.Format_RGB888)
            self.label_name.setPixmap(QPixmap.fromImage(qImg))
            
            self.label_name_objects.append(self.label_name)
            #Authorization Status
            self.authorization_label_name = QtWidgets.QLabel(enrolement_ChangeAccess)
            self.authorization_label_name.setGeometry(15 + column * 185, 215 + row* 280, 170, 40)
            self.authorization_label_name.setStyleSheet("font-size:16px;"
                                              "font-family: Play;")
            self.authorization_label_name.setAlignment(QtCore.Qt.AlignCenter)
            
            
            self.authorization_label_name_objects.append(self.authorization_label_name)
            #Remove Access Button
            self.remove_button_name = QtWidgets.QPushButton(enrolement_ChangeAccess)
            self.remove_button_name.setGeometry(23 + column * 185, 260+ row* 280, 153, 40)
            self.remove_button_name.setText("Remove Access")
            self.remove_button_name.setIcon(QIcon("./imgs/delete.png"))
            self.remove_button_name.setIconSize(QtCore.QSize(20, 20))
            self.remove_button_name.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
            )
            self.remove_button_name.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            self.remove_button_name.setVisible(False)
            self.remove_button_name.clicked.connect(partial(self.Remove,option,  user_number-1))

            
            #Grant Access Button
            self.grant_button_name = QtWidgets.QPushButton(enrolement_ChangeAccess)
            self.grant_button_name.setGeometry(23 + column * 185, 260 + row* 280, 153, 40)
            self.grant_button_name.setText("Grant Access")
            self.grant_button_name.setIcon(QIcon("./imgs/add.png"))
            self.grant_button_name.setIconSize(QtCore.QSize(20, 20))
            self.grant_button_name.setVisible(True)
            self.grant_button_name.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
            )
            self.grant_button_name.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            self.grant_button_name.clicked.connect(partial(self.Grant,option,  user_number-1))

            if option in authorizations:
                self.authorization_label_name.setText(option + '\n Authorized')
                self.remove_button_name.setVisible(True)
                self.grant_button_name.setVisible(False)
            else: 
                self.authorization_label_name.setText(option + '\n Unauthorized')
                self.remove_button_name.setVisible(False)
                self.grant_button_name.setVisible(True)
            
            self.remove_button_name_objects.append(self.remove_button_name)
            self.grant_button_name_objects.append(self.grant_button_name)

            user_number+=1

        print(self.label_name_objects)
        print(self.authorization_label_name_objects)
        print(self.remove_button_name_objects)
        print(self.grant_button_name_objects)

    def Grant(self,person, user_number):
        print(user_number)
 
        authorizations = np.load(HISTORY_PATH+'authorized.npy')
        authorizations = np.append(authorizations, str(person))
        print(authorizations)
        np.save(HISTORY_PATH+'authorized.npy', authorizations)
        self.authorization_label_name_objects[user_number].setText("{}\n Authorized".format(person))
        self.authorization_label_name_objects[user_number].setStyleSheet("color: #ff4b3c;"
                                              "font-size: 16px;")
        self.remove_button_name_objects[user_number].setVisible(True)
        self.grant_button_name_objects[user_number].setVisible(False)


    def Remove(self, person, user_number):
        print(user_number)

        authorizations = np.load(HISTORY_PATH+'authorized.npy')
        index = np.where(authorizations == person)[0]
        print(index)
        if person in authorizations:
            authorizations = np.delete(authorizations, index)
            np.save(HISTORY_PATH+'authorized.npy', authorizations)
            self.authorization_label_name_objects[user_number].setText("{}\n Unauthorized ".format(person))
            self.authorization_label_name_objects[user_number].setStyleSheet("color: #ff4b3c;"
                                                  "font-size: 16px;"
)
            print(authorizations)
        self.remove_button_name_objects[user_number].setVisible(False)
        self.grant_button_name_objects[user_number].setVisible(True)

    def retranslateUi(self, enrolement_ChangeAccess):
        _translate = QtCore.QCoreApplication.translate
        enrolement_ChangeAccess.setWindowTitle(_translate("enrolement_ChangeAccess", "FRekoAccess - Change Access"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    enrolement_ChangeAccess = QtWidgets.QMainWindow()
    ui = Ui_enrolement_ChangeAccess()
    ui.setupUi(enrolement_ChangeAccess)
    enrolement_ChangeAccess.show()
    sys.exit(app.exec_())
