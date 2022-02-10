
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

from paths import *

class Ui_enrolement_ChangeAccess(object):
    def setupUi(self, enrolement_ChangeAccess):
        enrolement_ChangeAccess.setObjectName("enrolement_ChangeAccess")
        enrolement_ChangeAccess.setFixedSize(900, 830)
        enrolement_ChangeAccess.setStyleSheet("background-color: #262626;"
        "color: #FFFFFF;"
        "font-family: Titillium;"
        "font-size: 18px;")
        enrolement_ChangeAccess.setWindowIcon(QIcon("./imgs/exia_logo.jpg"))
        self.centralwidget = QtWidgets.QWidget(enrolement_ChangeAccess)
        self.centralwidget.setObjectName("centralwidget")


        self.label = QtWidgets.QLabel(enrolement_ChangeAccess)
        self.label.setGeometry(60,40,560,450)
        self.label.setStyleSheet("border: 1px solid white;")

    def retranslateUi(self, enrolement_ChangeAccess):
        _translate = QtCore.QCoreApplication.translate
        enrolement_ChangeAccess.setWindowTitle(_translate("enrolement_ChangeAccess", "Add New Person"))
        self.label.setText(_translate("enrolement_ChangeAccess", "TEST"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    enrolement_addNew = QtWidgets.QMainWindow()
    ui = Ui_enrolement_ChangeAccess()
    ui.setupUi(enrolement_addNew)
    enrolement_addNew.show()
    sys.exit(app.exec_())
