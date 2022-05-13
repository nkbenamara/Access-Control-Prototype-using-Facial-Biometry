
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
        enrolement_ChangeAccess.setFixedSize(900, 250)
        enrolement_ChangeAccess.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        enrolement_ChangeAccess.setWindowIcon(QIcon("./imgs/logo.jpg"))

        #Central Widget
        self.centralwidget = QtWidgets.QWidget(enrolement_ChangeAccess)
        self.centralwidget.setObjectName("centralwidget")
        # Create combobox and add items.


        #Selected User Picture
        self.face_pic = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.face_pic.setGeometry(15,40,170,170)
        self.face_pic.setStyleSheet("border: 1px solid white;")
        self.face_pic.setText("Selected \n User")
        self.face_pic.setFont(font7)
        self.face_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.face_pic.setScaledContents(True)

        #Class List
        self.comboBox = QComboBox(enrolement_ChangeAccess)
        self.comboBox.setGeometry(568, 85, 200, 30)
        self.comboBox.setObjectName(("comboBox"))
        self.comboBox.addItem("Select user...")
        self.comboBox.setStyleSheet(
            "border-width: 2px;"
            "border-radius: 10px;" 
            "border-color: red;"
            "background-color: #262626;"
             "}"
             
        "QComboBox QAbstractItemView{"
             "border: 2px;color:#BB86FC;"
             "selection-background-color: #BB86FC;"
             "}"
        "QComboBox::item: selected"
        "{"
            "background-color:rgb(170, 170, 255);"
            "color: #BB86FC;"
        "}")

        options = load_actual_collaborators("vgg16")
        for option in options:
            self.comboBox.addItem(option)

        #Authorization Status
        self.autorization_label = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.autorization_label.setGeometry(215, 80, 250, 50)
        self.autorization_label.setStyleSheet("font-size:20px;"
                                              "font-family: Play;")

        #Edit Button                                      
        self.editButton = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.editButton.setGeometry(600, 150, 142, 40)
        self.editButton.setText("Edit")
        self.editButton.setIcon(QIcon("./imgs/edit.png"))
        self.editButton.setIconSize(QtCore.QSize(30, 30))
        self.editButton.setStyleSheet("QPushButton""{"
                                "background-color: #262626"
                                      "}"
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.editButton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.editButton.clicked.connect(self.editPersonnel)

        #Remove Access Button
        self.remove_access = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.remove_access.setGeometry(215, 120, 153, 40)
        self.remove_access.setText("Remove Access")
        self.remove_access.setIcon(QIcon("./imgs/delete.png"))
        self.remove_access.setIconSize(QtCore.QSize(20, 20))
        self.remove_access.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.remove_access.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.remove_access.setVisible(False)
        self.remove_access.clicked.connect(self.Remove)

        #Grant Access Button
        self.grant_access = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.grant_access.setGeometry(215, 120, 153, 40)
        self.grant_access.setText("Grant Access")
        self.grant_access.setIcon(QIcon("./imgs/add.png"))
        self.grant_access.setIconSize(QtCore.QSize(20, 20))
        self.grant_access.setVisible(False)
        self.grant_access.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.grant_access.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.grant_access.clicked.connect(self.Grant)

        

    def editPersonnel(self):
        person = self.comboBox.currentText()
        path=glob(MAIN_PATH + "/dataset_cropped/"+ person +"/*")[0]
        
        print(path)
        face_img = cv2.imread(path)
        height, width, channel = face_img.shape
        step = channel * width
        y_train = np.load(GALLERY_PATH+'vgg16_y_train.npy')
        authorizations = np.load(HISTORY_PATH+'authorized.npy')
        if os.path.isdir(path) == False:
            print("user not selected")
            self.autorization_label.setText("Please select a user!")
        if person in y_train:
            if person in authorizations:
                self.autorization_label.setText("{} is authorized".format(person))
                self.autorization_label.setStyleSheet("color: #ff4b3c;"
                                                      "font-size: 18px;")
                self.remove_access.setVisible(True)
                self.grant_access.setVisible(False)
            else:
                self.autorization_label.setText("{} is unauthorized".format(person))
                self.autorization_label.setStyleSheet("color: #ff4b3c;"
                                                      "font-size: 18px;")
                self.autorization_label.setAlignment(QtCore.Qt.AlignLeft)
                self.remove_access.setVisible(False)
                self.grant_access.setVisible(True)

        else:
            self.autorization_label.setText("{} is not part of the personnel".format(person))
            self.autorization_label.setStyleSheet("color: #FFFFFF;"
                                                  "font-size: 18px")
            self.remove_access.setVisible(False)
            self.grant_access.setVisible(False)

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        qImg = QImage(face_img.data, width, height, step, QImage.Format_RGB888)
        self.face_pic.setPixmap(QPixmap.fromImage(qImg))

    def Grant(self):
        person = self.comboBox.currentText()
        authorizations = np.load(HISTORY_PATH+'authorized.npy')
        authorizations = np.append(authorizations, str(person))
        print(authorizations)
        np.save(HISTORY_PATH+'authorized.npy', authorizations)
        self.autorization_label.setText("{} is now authorized".format(person))
        self.autorization_label.setStyleSheet("color: #ff4b3c;"
                                              "font-size: 18px;")
        self.remove_access.setVisible(True)
        self.grant_access.setVisible(False)


    def Remove(self):
        person = self.comboBox.currentText()
        authorizations = np.load(HISTORY_PATH+'authorized.npy')
        index = np.where(authorizations == person)[0]
        print(index)
        if person in authorizations:
            authorizations = np.delete(authorizations, index)
            np.save(HISTORY_PATH+'authorized.npy', authorizations)
            self.autorization_label.setText("{} is now unauthorized ".format(person))
            self.autorization_label.setStyleSheet("color: #ff4b3c;"
                                                  "font-size: 18px;"
)
            print(authorizations)
        self.remove_access.setVisible(False)
        self.grant_access.setVisible(True)

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
