
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil

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
        enrolement_ChangeAccess.setObjectName("enrolement_ChangeAccess")
        enrolement_ChangeAccess.setFixedSize(900, 830)
        enrolement_ChangeAccess.setStyleSheet("background-color: #121212;" 
        "color: #FFFFFF;"
        "font-family: Titillium;"
        "font-size: 18px;")
        enrolement_ChangeAccess.setWindowIcon(QIcon("./imgs/exia_logo.jpg"))


        self.centralwidget = QtWidgets.QWidget(enrolement_ChangeAccess)
        self.centralwidget.setObjectName("centralwidget")
        # Create combobox and add items.



        self.face_pic = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.face_pic.setGeometry(15,40,475,400)
        self.face_pic.setStyleSheet("border: 1px solid white;")
        self.face_pic.setText("PICTURE OF SELECTED USER")
        self.face_pic.setScaledContents(True)


        #self.vgg16_enroled_list = QtWidget
        # s.QLabel(enrolement_ChangeAccess)
        #self.vgg16_enroled_list.setGeometry(475, 40, 425, 350)
        #self.vgg16_enroled_list.setStyleSheet("border: 1px solid white;")
        #self.vgg16_enroled_list.setText("")

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
        #self.nameText = QtWidgets.QLineEdit(enrolement_ChangeAccess)


        #self.nameText.setGeometry(40, 475, 200, 30)
        #self.nameText.setStyleSheet("border: 1px solid white;"
                      #"font-style:bold;"
                      #"background-color:rgb(255,255,255);"
                      #"color: rgb(0,0,0);"
                      #)
        self.CURRENT_PATH = os.getcwd()
        self.nameLabel = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.nameLabel.setGeometry(QtCore.QRect(560, 40, 255, 30))
        self.nameLabel.setStyleSheet(
                      "font-style:bold;"
                      "font-size: 22px;"    
                      "background-color: #121212;"
                      "color: #BB86FC;"
                      )
        self.nameLabel.setText("Choose the person to edit:")
        self.nameButton = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.nameButton.setGeometry(600, 150, 150, 30)
        self.nameButton.setText("EDIT")
        self.nameButton.setStyleSheet("QPushButton""{"
                                "background-color: #3700B3"
                                      "}"
                                "QPushButton::hover"
                             "{"
                             "background-color: #BB86FC;"
                            "color: #FFFFFF;"
                             "}"
                            "border: 1px solid white;"
                            )
        self.nameButton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.nameButton.clicked.connect(self.editPersonnel)
        self.autorization_label = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.autorization_label.setGeometry(50, 500, 750, 275)
        self.autorization_label.setText("    AUTHORIZATION STATUS")
        self.autorization_label.setStyleSheet("font-size:50px;"
                                              "font-style: roboto;")
                                              #"border: 1px solid white;"
        self.remove_access = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.remove_access.setGeometry(510, 250, 175, 45)
        self.remove_access.setText("Remove access")
        self.remove_access.setStyleSheet(
            "QPushButton"
        "{"
            "background-color: #CF6679;"    
             "border-style: outset;"
             "border-width: 2px;"
              "border-radius: 10px;"
                "border-color: gray;"
            "font-style: roboto;"
            "font-size: 20px;"
        "}"
            
            "QPushButton::hover"
            "{"
            "background-color: #B00020;"
            "color: black;"
            "}"
            "QPushButton::pressed" "{" "background-color: red" "}"
        )
        self.remove_access.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.remove_access.setVisible(False)

        self.grant_access = QtWidgets.QPushButton(enrolement_ChangeAccess)
        self.grant_access.setGeometry(700, 250, 175, 45)
        self.grant_access.setText("Grant access")
        self.grant_access.setVisible(False)
        self.grant_access.setStyleSheet(
             "QPushButton"
        "{"
            "background-color: #018786;"
             "border-style: outset;"
             "border-width: 2px;"
              "border-radius: 10px;"
                "border-color: gray;"
               "font-style: roboto;"
            "font-size: 20px;"
        "}"
             "QPushButton::hover"
             "{"
             "background-color:#03DAC5;"
             "color: black;"
             "}"
             "QPushButton::pressed" "{" "background-color: green" "}"
        )
        self.grant_access.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        self.grant_access.clicked.connect(self.Grant)
        self.remove_access.clicked.connect(self.Remove)
    def editPersonnel(self):
        person = self.comboBox.currentText()
        rand_pic = random.randrange(0, 50)
        path = self.CURRENT_PATH + "/dataset_cropped/"+ person +"/{}.jpg".format(rand_pic)
        print(path)
        face_img = cv2.imread(path)
        height, width, channel = face_img.shape
        step = channel * width
        y_train = np.load('./gallery/vgg16_y-train.npy')
        authorizations = np.load('./gallery/authorized.npy')
        if os.path.isdir(path) == False:
            print("user not selected")
            self.autorization_label.setText("Please select a user!")
        if person in y_train:
            if person in authorizations:
                self.autorization_label.setText("Authorization for {} is".format(person) +'\n'+"\t:AUTHORIZED!")
                self.autorization_label.setStyleSheet("color: #03DAC6;"
                                                      "font-size: 50px;"
                                                      "font-weight: bold;"
                                                      "font-style: roboto;")
            else:
                self.autorization_label.setText("Authorizations for {} is".format(person) + '\n' + "\t:!UNAUTHORIZED!")
                self.autorization_label.setStyleSheet("color: #CF6679;"
                                                      "font-size: 50px;"
                                                      "font-weight: bold;"
                                                      "font-style: roboto;")
                self.autorization_label.setAlignment(QtCore.Qt.AlignLeft)
            self.remove_access.setVisible(True)
            self.grant_access.setVisible(True)

        else:
            self.autorization_label.setText("{} is not part of the personnel".format(person))
            self.autorization_label.setStyleSheet("color: #FFFFFF;"
                                                  "font-size: 50px"
                                                  "font-weight: bold;"
                                                  "font-style: roboto;")
            self.remove_access.setVisible(False)
            self.grant_access.setVisible(False)

        qImg = QImage(face_img.data, width, height, step, QImage.Format_RGB888)
        self.face_pic.setPixmap(QPixmap.fromImage(qImg))

    def Grant(self):
        person = self.comboBox.currentText()
        authorizations = np.load('./gallery/authorized.npy')
        authorizations = np.append(authorizations, str(person))
        print(authorizations)
        np.save('./gallery/authorized.npy', authorizations)
        self.autorization_label.setText("Authorizations for {} has been changed".format(person) + '\n' + ": This person is now AUTHORIZED!" +'\n')
        self.autorization_label.setStyleSheet("color: #03DAC6;"
                                              "font-size: 35px;"
                                              "font-weight: bold;"
                                              "font-style: roboto;")


    def Remove(self):
        person = self.comboBox.currentText()
        authorizations = np.load('./gallery/authorized.npy')
        index = np.where(authorizations == person)[0]
        print(index)
        if person in authorizations:
            authorizations = np.delete(authorizations, index)
            np.save('./gallery/authorized.npy', authorizations)
            self.autorization_label.setText("Authorizations for {} has been changed".format(person) + '\n' + ": This person is now !UNAUTHORIZED!"+'\n')
            self.autorization_label.setStyleSheet("color: #CF6679;"
                                                  "font-size: 35px;"
                                                  "font-weight: bold;"
                                                  "font-style: roboto;")
            print(authorizations)

    def retranslateUi(self, enrolement_ChangeAccess):
        _translate = QtCore.QCoreApplication.translate
        enrolement_ChangeAccess.setWindowTitle(_translate("enrolement_ChangeAccess", "FRekoAccess: Change Access"))
        self.vgg16_enroled_list.setText(_translate("enrolement_ChangeAccess", ""))
        self.nameLabel.setText(_translate("enrolement_ChangeAccess", ""))
        self.nameText.setText(_translate("enrolement_ChangeAccess", " "))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    enrolement_addNew = QtWidgets.QMainWindow()
    ui = Ui_enrolement_ChangeAccess()
    ui.setupUi(enrolement_addNew)
    enrolement_addNew.show()
    sys.exit(app.exec_())
