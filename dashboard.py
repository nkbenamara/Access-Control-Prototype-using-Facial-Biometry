from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon
from glob import glob
import numpy as np
from paths import *

from utils import donutGenerator
class Ui_Dashboard_Ui(object):
    def setupUi(self, Dashboard_Ui):
        '''
        MAIN WINDOW - DASHBOARD
        '''
        #Define Fonts
        QtGui.QFontDatabase.addApplicationFont("./fonts/Play-Regular.ttf")

        Dashboard_Ui.setObjectName("Dashboard_Ui")
        Dashboard_Ui.setFixedSize(900, 830)
        Dashboard_Ui.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        Dashboard_Ui.setWindowIcon(QIcon("./imgs/logo.jpg"))
        self.centralwidget = QtWidgets.QWidget(Dashboard_Ui)
        self.centralwidget.setObjectName("centralwidget")

        #Collaboratos
        self.class_number = QtWidgets.QLabel(self.centralwidget)
        self.class_number.setGeometry(QtCore.QRect(30, 40, 400, 400))
        self.class_number.setObjectName("class_number")

        self.collabs = QtWidgets.QLabel(self.centralwidget)
        self.collabs.setGeometry(QtCore.QRect(30, 500, 400, 220))
        self.collabs.setObjectName("collabs")

        #Donut1
        self.donut1 = QtWidgets.QLabel(self.centralwidget)
        self.donut1.setGeometry(QtCore.QRect(485, 40, 220, 100))
        self.donut1.setObjectName("donut1")
        self.donut1.setStyleSheet("border: 1px solid #4B0082;""font-style:roboto;")

        self.donut1_title = QtWidgets.QLabel(self.centralwidget)
        self.donut1_title.setGeometry(QtCore.QRect(485, 250, 400, 25))
        self.donut1_title.setStyleSheet("color: white;")
        self.donut1_title.setText("Facial Data Distribution per Person")
        self.donut1_title.setAlignment(QtCore.Qt.AlignCenter)

        #Donut2
        self.donut2 = QtWidgets.QLabel(self.centralwidget)
        self.donut2.setGeometry(QtCore.QRect(485, 285, 220, 100))
        self.donut2.setObjectName("donut2")
        self.donut2.setStyleSheet("border: 1px solid #4B0082;""font-style:roboto;")

        self.donut2_title = QtWidgets.QLabel(self.centralwidget)
        self.donut2_title.setGeometry(QtCore.QRect(485, 495, 400, 25))
        self.donut2_title.setStyleSheet("color: white;")
        self.donut2_title.setText("Distribution of Granted & Denied Access")
        self.donut2_title.setAlignment(QtCore.Qt.AlignCenter)

        #Donut3
        self.donut3 = QtWidgets.QLabel(self.centralwidget)
        self.donut3.setGeometry(QtCore.QRect(485, 525, 220, 100))
        self.donut3.setObjectName("donut3")
        self.donut3.setStyleSheet("border: 1px solid #4B0082;""font-style:roboto;")

        self.donut3_title = QtWidgets.QLabel(self.centralwidget)
        self.donut3_title.setGeometry(QtCore.QRect(485, 735, 400, 25))
        self.donut3_title.setStyleSheet("color: white;")
        self.donut3_title.setText("Distribution of Day/Night Access Time")
        self.donut3_title.setAlignment(QtCore.Qt.AlignCenter)
    
        Dashboard_Ui.setCentralWidget(self.centralwidget)

        self.retranslateUi(Dashboard_Ui)
        QtCore.QMetaObject.connectSlotsByName(Dashboard_Ui)
        donutGenerator(GALLERY_PATH+"y_train_camera1.npy",STAT_PATH+"img.png")
        donutGenerator(HISTORY_PATH+"access_history.npy",STAT_PATH+"img2.png")
        donutGenerator(HISTORY_PATH+"accessTime_history.npy",STAT_PATH+"img3.png")
        self.classCount()
        self.pieImg()


    def classCount(self):
        y_train = np.load(GALLERY_PATH+"y_train_camera1.npy")

        #data = np.unique(y_train, return_counts=True)[1]
        class_count = np.shape(np.unique(y_train, return_counts=True)[0])[0]

        self.class_number.setText("{}".format(class_count))
        self.class_number.setStyleSheet("font-size: 300px;""font-style: roboto;""text-align: center;""color: white;")
        self.class_number.setAlignment(QtCore.Qt.AlignCenter)

        self.collabs.setText("Enrolled\nPersonnel")
        self.collabs.setStyleSheet("font-size: 85px;" "font-style: roboto;""color: white;")
        self.collabs.setAlignment(QtCore.Qt.AlignCenter)


    def pieImg(self):
        pixmap1 = QPixmap(STAT_PATH+'img.png')
        pixmap2 = QPixmap(STAT_PATH+'img2.png')
        pixmap3 = QPixmap(STAT_PATH+'img3.png')
        self.donut1.setPixmap(pixmap1)
        self.donut1.resize(pixmap1.width(), pixmap1.height())
        self.donut2.setPixmap(pixmap2)
        self.donut2.resize(pixmap2.width(), pixmap2.height())
        self.donut3.setPixmap(pixmap3)
        self.donut3.resize(pixmap3.width(), pixmap3.height())


    def retranslateUi(self, Dashboard_Ui):
        _translate = QtCore.QCoreApplication.translate
        Dashboard_Ui.setWindowTitle(_translate("Dashboard_Ui", "Dashboard"))
        self.donut1.setText(_translate("Dashboard_Ui", ""))
        self.donut2.setText(_translate("Dashboard_Ui", ""))
        self.donut3.setText(_translate("Dashboard_Ui", ""))
        self.collabs.setText(_translate("Dashboard_Ui", ""))
        self.class_number.setText(_translate("Dashboard_Ui", ""))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dashboard_Ui = QtWidgets.QMainWindow()
    ui = Ui_Dashboard_Ui()
    ui.setupUi(Dashboard_Ui)
    Dashboard_Ui.show()
    sys.exit(app.exec_())
