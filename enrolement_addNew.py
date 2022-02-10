
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

class Ui_enrolement_addNew(object):
    def setupUi(self, enrolement_addNew):
        enrolement_addNew.setObjectName("enrolement_addNew")
        enrolement_addNew.resize(900, 830)
        enrolement_addNew.setStyleSheet("background-color: #262626;"
        "color: #FFFFFF;"
        "font-family: Titillium;"
        "font-size: 18px;")
        enrolement_addNew.setWindowIcon(QIcon("./imgs/exia_logo.jpg"))
        self.centralwidget = QtWidgets.QWidget(enrolement_addNew)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QLabel(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(420, 15, 451, 441))
        font = QtGui.QFont()
        font.setPointSize(21)
        self.frame.setFont(font)
        self.frame.setObjectName("frame")
        self.frame.setStyleSheet("border: 1px solid white;"
                                     "font-style:bold;"
                                     )
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(700, 525, 65, 21))
        self.spinBox.setObjectName("spinBox")

        self.nbFrames = QtWidgets.QLabel(self.centralwidget)
        self.nbFrames.setGeometry(QtCore.QRect(625, 475, 330, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.nbFrames.setFont(font)
        self.nbFrames.setObjectName("nbFrames")

        self.take_pic = QtWidgets.QPushButton(self.centralwidget)
        self.take_pic.setGeometry(QtCore.QRect(650, 590, 150, 30))
        self.take_pic.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.take_pic.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.take_pic.setObjectName("take_pic")

        self.nameText = QtWidgets.QLineEdit(self.centralwidget)
        self.nameText.setGeometry(QtCore.QRect(640, 675, 200, 30))
        self.nameText.setStyleSheet("border: 1px solid white;"
                      "font-style:bold;"
                      "background-color:rgb(255,255,255);"
                                    "color: rgb(0,0,0);"
                      )


        self.nameLabel = QtWidgets.QLabel(self.centralwidget)
        self.nameLabel.setGeometry(QtCore.QRect(640, 635, 220, 30))

        self.startBTN = QtWidgets.QPushButton(self.centralwidget)
        self.startBTN.setGeometry(QtCore.QRect(425, 500, 170, 51))
        font = QtGui.QFont()
        font.setPointSize(23)
        self.startBTN.setFont(font)
        QToolTip.setFont(QFont('SansSerif', 10))
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
        self.startBTN.setToolTip("Start Face Recognition")  # Tool tip
        self.startBTN.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.startBTN.setObjectName("startBTN")

        self.enrolement_mode = QtWidgets.QLabel(self.centralwidget)
        self.enrolement_mode.setGeometry(QtCore.QRect(40, 50, 300, 21))
        self.enrolement_mode.setFont(font)
        self.enrolement_mode.setObjectName("enrolement_mode")
        self.enrolement_mode.setStyleSheet("font-style: roboto;"
                                           "font-size: 0.95em;"
                                           "font-weight: bold;"
                                           "color: green;")
        self.radio_vgg16 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_vgg16.setGeometry(QtCore.QRect(30, 100, 121, 21))

        font.setPointSize(12)
        self.radio_vgg16.setFont(font)
        self.radio_vgg16.setObjectName("radio_vgg16")
        self.radio_resnet50 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_resnet50.setGeometry(QtCore.QRect(180, 100, 121, 21))

        font.setPointSize(12)
        self.radio_resnet50.setFont(font)
        self.radio_resnet50.setObjectName("radioButton_2")

        self.authorisations_label = QtWidgets.QLabel(self.centralwidget)
        self.authorisations_label.setGeometry(40, 150, 220, 40)
        self.authorisations_label.setStyleSheet("""font-style: bold;
                                                    font-family: roboto;
                                                    font-weight: bold;
                                                    font-size: 0.85em;
                                                    color: green;""")


        self.radio_authorized = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_authorized.setGeometry(QtCore.QRect(30, 185, 121, 21))
        font.setPointSize(12)
        self.radio_authorized.setFont(font)
        self.radio_authorized.setObjectName("radio_vgg16")
        self.btngroup1 = QButtonGroup()
        self.btngroup1.addButton(self.radio_vgg16)
        self.btngroup1.addButton(self.radio_resnet50)


        self.radio_unauthorized = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_unauthorized.setGeometry(QtCore.QRect(180, 185, 121, 21))
        font.setPointSize(12)
        self.radio_unauthorized.setFont(font)
        self.radio_unauthorized.setObjectName("radio_vgg16")
        self.btngroup2 = QButtonGroup()
        self.btngroup2.addButton(self.radio_authorized)
        self.btngroup2.addButton(self.radio_unauthorized)
        self.enrolementBTN = QtWidgets.QPushButton(self.centralwidget)
        self.enrolementBTN.setGeometry(QtCore.QRect(70, 225, 171, 35))
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
        self.startBTN.clicked.connect(self.viewCam)
        self.enrolementBTN.clicked.connect(self.feature_extraction)
        self.take_pic.clicked.connect(self.capture_image)

    def viewCam(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_DETECTION_MODELS+'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(EYE_DETECTION_MODELS+'haarcascade_eye.xml')
        frame_rate = 10
        prev = 0


        while True:
            stime = time.time()
            time_elapsed = time.time() - prev
            # Capture frame-by-frame
            ret, frame = self.capture.read()
            if time_elapsed > 1. / frame_rate:
                prev = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = self.face_cascade.detectMultiScale(gray)
            height, width, channel = image.shape
            step = channel * width

            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                # comparer mon vecteur avec mon embedding
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = frame[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)#detect eyes inside face rectangle
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ ew, ey + eh), (0,0,255), 2)


            self.frame.setPixmap(QPixmap.fromImage(qImg))


            #self.show_fps.setText('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(0) & 0xFF == ord('z'):
                break

        self.capture.release()
        cv2.destroyAllWindows()




    def capture_image(self):
        face_cascade = cv2.CascadeClassifier(FACE_DETECTION_MODELS+'haarcascade_frontalface_default.xml')
        ret, frame = self.capture.read()
        directory = self.nameText.text()
        path = GALLERY_IMAGES_PATH+ str(directory) + "/"
        if os.path.isdir(path):
            print("Path Already Exists...")
            print("Removing " + str(directory))
            shutil.rmtree(GALLERY_IMAGES_PATH+"{}".format(str(directory)), ignore_errors=True)
            print(str(directory) + 'Has been removed')
            print("Creating New folder named : " + str(directory))
            os.mkdir(path)
        else:
            os.mkdir(path)

        frame_number = self.spinBox.value()
        self.nameText.setEnabled(False)
        #N = 5
        #addon = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        count = 0
        while ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    captured_img = cv2.imwrite(path + "frame-{}.jpg".format(count)  , roi_color)
                    ret, frame = self.capture.read()
                    count += 1
                    if count == 25 or count == 40:
                        self.notif.setText("Please change position")
                        self.notif.setStyleSheet("""color : white;""")

                    print("Picture " + str(count) + " Saved successfully")
            except:
                pass
            if count == frame_number:
                break
        self.viewCam()



    def feature_extraction(self):
        radio_vgg = self.radio_vgg16.isChecked()
        radio_resnet = self.radio_resnet50.isChecked()
        grant = self.radio_authorized.isChecked()
        deny = self.radio_unauthorized.isChecked()
        name = self.nameText.text()
        image_list = []
        feature_vectors = []
        labels = []
        images = glob(GALLERY_IMAGES_PATH + str(name) + "/"+ "*.jpg")
        file_x_train = GALLERY_PATH+'vgg16_x_train.npy'
        file_y_train = GALLERY_PATH+'vgg16_y_train.npy'
        x_train = np.load(file_x_train)
        y_train = np.load(file_y_train)
        authorized = np.load(HISTORY_PATH+"authorized.npy")
        unauthorized = np.load(HISTORY_PATH+"unauthorized.npy")
        if radio_vgg == True:
            model_type = "vgg16"
            preprocessing = 1
        else:
            model_type = "resnet50"
            preprocessing = 2


        vgg_face_model = VGGFace(model=model_type, include_top=False, input_shape=(224, 224, 3), pooling='avg')
        for image in images:
            roi_color = cv2.imread(image)
            roi_color = cv2.resize(roi_color, (224, 224),interpolation=cv2.INTER_AREA)  # load an image and resize it to 224,224 like vgg face input size
            roi_color = img_to_array(roi_color)  # convert the image to an array
            roi_color = np.expand_dims(roi_color,  axis=0)
            roi_color = utils.preprocess_input(roi_color, version= preprocessing)  # preprocess the image 1 = vggface resnet = 2)
            feature_vector = vgg_face_model.predict(roi_color)  # extract the features
            np.shape(feature_vector)
            feature_vectors.append(feature_vector)#append the current feature vector to the gallery
            labels.append(str(name))#append the current label to the gallery


        feature_vectors = np.squeeze(np.array(feature_vectors), axis=1)
        labels = np.expand_dims(np.array(labels), axis=1)
        #exception si fichier n'existe pas np.save else concatenate
        feature_vectors = np.concatenate((x_train, feature_vectors), axis=0)
        labels = np.concatenate((y_train, labels), axis=0)
        data_filename = GALLERY_PATH + str(model_type) + '_x_train.npy'      #datetime.now().strftime('%Y_%m_%d-at-%H_%M_%S.npy'
        labels_filename = GALLERY_PATH + str(model_type) + '_y_train.npy'   #datetime.now().strftime('%Y_%m_%d-at-%H_%M_%S.npy'
        np.save(data_filename, feature_vectors)
        np.save(labels_filename, labels)

        if grant == True:
            authorized = np.append(authorized, name)
            print(authorized)
        if deny == True:
            unauthorized = np.append(unauthorized, name)
            print('pring thieos')
        print('Saved to : {}/{}'.format(GALLERY_PATH, data_filename))
        print('Saved to : {}/{}'.format(GALLERY_PATH, labels_filename))
        self.viewCam()
        #save_vector = np.concatenate((gallery, file_x_train), axis=1)





    def retranslateUi(self, enrolement_addNew):
        _translate = QtCore.QCoreApplication.translate
        enrolement_addNew.setWindowTitle(_translate("enrolement_addNew", "Add New Person"))
        self.frame.setText(_translate("enrolement_addNew", "         FRAME CAPTURE"))
        self.nbFrames.setText(_translate("enrolement_addNew", "Nb of Frames to be captured"))
        self.take_pic.setText(_translate("enrolement_addNew", "Take Pictures"))
        self.startBTN.setText(_translate("enrolement_addNew", "START CAPTURE"))
        self.enrolement_mode.setText(_translate("enrolement_addNew", "ENROLEMENT MODE SELECTION :"))
        self.enrolementBTN.setText(_translate("enrolement_addNew", "Start Scan"))
        self.radio_vgg16.setText(_translate("enrolement_addNew", "VGG16"))
        self.radio_resnet50.setText(_translate("enrolement_addNew", "RESNET 50"))
        self.radio_authorized.setText(_translate("enrolement_addNew", "Authorized"))
        self.radio_unauthorized.setText(_translate("enrolement_addNew", "Unauthorized"))
        self.nameLabel.setText(_translate("enrolement_addNew", "Name of the person:"))
        self.notif.setText(_translate("enrolement_addNew", ""))
        self.authorisations_label.setText(_translate("enrolement_addNew", "Authorisation Setting :"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    enrolement_addNew = QtWidgets.QMainWindow()
    ui = Ui_enrolement_addNew()
    ui.setupUi(enrolement_addNew)
    enrolement_addNew.show()
    sys.exit(app.exec_())
