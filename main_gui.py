from enrolement_addNew import Ui_enrolement_addNew
from enrolement_ChangeAccess import Ui_enrolement_ChangeAccess
from dashboard import Ui_Dashboard_Ui
from utils import generate_report
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QCursor, QIcon, QFont
from PyQt5.QtWidgets import QToolTip

import sys
from datetime import datetime
import os
import cv2
import random
import string
from glob import glob
import numpy as np
import time
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras_vggface import utils
from utils import prediction_cosine_similarity2, eye_aspect_ratio, delete_personel, reinteger_all, convert_and_trim_bb

from imutils import face_utils
from paths import *
from models import *

enrol = Ui_enrolement_addNew()



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        '''
        MAIN WINDOW
        '''
        
        #Define Fonts
        QtGui.QFontDatabase.addApplicationFont("./fonts/Play-Regular.ttf")
        
        ##Font7 & Font15
        font7 = QtGui.QFont("Play", 7)
        font15 = QtGui.QFont("Play",15)

        #Main Window
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(900, 830)
        MainWindow.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
        MainWindow.setWindowIcon(QIcon("./imgs/logo.jpg"))

        #Central Widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #Display Live FPS
        self.show_fps = QtWidgets.QLabel(self.centralwidget)
        self.show_fps.setGeometry(QtCore.QRect(750, 550, 75, 31))
        
        #Start Record Button
        self.recordBTN = QtWidgets.QPushButton(self.centralwidget)
        self.recordBTN.setGeometry(QtCore.QRect(500, 495, 142, 40))
  
        self.recordBTN.setFont(font7)
        self.recordBTN.setStyleSheet(
                                "QPushButton::hover"
                             "{"
                             "background-color:rgb(255,255,255);"
                            "color: black;"
                             "}"
                            "border: 1px solid white;"
                            )
        self.recordBTN.setIcon(QIcon("./imgs/start.ico"))
        self.recordBTN.setIconSize(QtCore.QSize(20, 20))
        self.recordBTN.setShortcut('Ctrl+R')
        self.recordBTN.setToolTip("Start Face Recognition")  # Tool tip
        self.recordBTN.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.recordBTN.setObjectName("recordBTN")
        self.recordBTN.clicked.connect(self.viewCam)
        
        #Take Picture Button
        self.take_pic = QtWidgets.QPushButton(self.centralwidget)
        self.take_pic.setGeometry(QtCore.QRect(650, 495, 142, 40))
        self.take_pic.setFont(font7)
        self.take_pic.setStyleSheet(
            "QPushButton::hover"
            "{"
            "background-color:rgb(255,255,255);"
            "color: black;"
            "}"
            "border: 1px solid white;"
        )
        self.take_pic.setIcon(QIcon("./imgs/cam.png"))
        self.take_pic.setIconSize(QtCore.QSize(30, 30))
        self.take_pic.setShortcut('Ctrl+R')
        self.take_pic.setToolTip("Take a picture of face")  # Tool tip
        self.take_pic.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.take_pic.setObjectName("takePic")
        self.take_pic.setEnabled(False)
        self.take_pic.clicked.connect(self.capture_image)

        #Camera Live Recording Area
        self.cam_label = QtWidgets.QLabel(self.centralwidget)
        self.cam_label.setGeometry(QtCore.QRect(420, 15, 451, 441))
        self.cam_label.setObjectName("cam_label")
        self.cam_label.setStyleSheet("border: 1px solid white;"
                                     "font-style:bold;"
                                     )
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)

        ##EAR Label Area - Camera Live Area
        self.EAR_label = QtWidgets.QLabel(self.centralwidget)
        self.EAR_label.setGeometry(QtCore.QRect(600, 18, 265, 21))
        self.EAR_label.setObjectName("EAR_label")
        self.EAR_label.setFont(font15)

        ##Blink Count Area - Camera Live Area
        self.blink_count = QtWidgets.QLabel(self.centralwidget)
        self.blink_count.setGeometry(QtCore.QRect(425, 18, 85, 21))
        self.blink_count.setObjectName("blink_count")
        self.blink_count.setFont(font15)

        #Taken Picture Area
        self.face_frame = QtWidgets.QLabel(self.centralwidget)
        self.face_frame.setGeometry(QtCore.QRect(30, 20, 360, 400))
        self.face_frame.setObjectName("face_frame")
        self.face_frame.setScaledContents(True)
        self.face_frame.setStyleSheet("border: 1px solid white;"
                       "font-style:bold;"
                       )
        self.face_frame.setAlignment(QtCore.Qt.AlignCenter)

        #Access Authorizations Area
        self.history = QtWidgets.QLabel(self.centralwidget)
        self.history.setGeometry(QtCore.QRect(30, 450, 360, 250))
        self.history.setObjectName("authorizations")
        self.history.setScaledContents(True)
        self.history.setStyleSheet("border: 1px solid white;"
                                      "font-style:bold;"
                                      )
        self.history.setAlignment(QtCore.Qt.AlignCenter)

        MainWindow.setCentralWidget(self.centralwidget)
        """
        MENU BAR
        """
        #Menu Bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setFont(font7)

        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("""
        QMenuBar
        {
            background-color: #0c0c13 ;
            color: #fff;
        }
        QMenuBar::item
        {
            font-style: Play;
            background-color: #0c0c13;
            color: #fff;
            padding-left : 5px; 
        }
        QMenuBar::item::selected
        {  
            background-color: #1b1553;
            color: #fff;
            
        }
        QMenu
        {   
            background-color: #1e0a30;
            color: #fff;
            padding-left : 5px; 
        }
        QMenu::item::selected
        {
            background-color: #1b1553;
            color: #fff;
            border : 1px solid #ff4b3c;
        }

        """)
        #Defining Menu Bar Options
        ##Face Recognition Menu
        self.menuFace_recognition = QtWidgets.QMenu(self.menubar)
        self.menuFace_recognition.setObjectName("menuFace_recognition")

        ##Enrolement Menu
        self.menuEnrolement = QtWidgets.QMenu(self.menubar)
        self.menuEnrolement.setObjectName("menuEnrolement")

        ##Dashboard Menu
        self.menuDashboard = QtWidgets.QMenu(self.menubar)
        self.menuDashboard.setObjectName("menuDashboard")

        ##Settings Menu
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        self.menuSettings.setStyleSheet("""QPushButton{background-color: lightgrey}""")

        MainWindow.setMenuBar(self.menubar)
        
        """
        ACTIONS - MENU BAR
        """
        #Face Recognition Action Option - Menu Bar
        self.menubar.addAction(self.menuFace_recognition.menuAction())

        #Enrollement Action Option - Menu Bar
        self.menubar.addAction(self.menuEnrolement.menuAction())

        #Dashboard Action Option - Menu Bar
        self.menubar.addAction(self.menuDashboard.menuAction())

        #Settings Action Option - Menu Bar
        self.menubar.addAction(self.menuSettings.menuAction())
        """
        ACTIONS - ENROLLEMENT MENU
        """
        #Add a Person Action to Enrolement Menu
        self.actionAjouter_personne = QtWidgets.QAction(MainWindow)
        self.actionAjouter_personne.setObjectName("actionAjouter_personne")
        self.actionAjouter_personne.setCheckable(True)
        self.menuEnrolement.addAction(self.actionAjouter_personne)

        #Add the Change Access Action to Enrolement Menu
        self.removeAccess = QtWidgets.QAction(MainWindow)
        self.removeAccess.setObjectName("removeAccess")
        self.removeAccess.setCheckable(True)
        self.menuEnrolement.addAction(self.removeAccess)
        
        self.menuEnrolement.triggered[QAction].connect(self.enrollment_windows)
        """
        ACTIONS - DASHBOARD MENU
        """
        #Dashboard Window Action to Dashboard Menu 
        self.dashboardWindow = QtWidgets.QAction(MainWindow)
        self.dashboardWindow.setObjectName("dashboardWindow")
        self.menuDashboard.addAction(self.dashboardWindow)

        self.menuDashboard.triggered[QAction].connect(self.openDashboard)

        """
        ACTIONS - SETTINGS MENU
        """
        #Add Dark/Light Mode to Settings Menu
        self.darkMode = QtWidgets.QAction(self.menuSettings)
        self.darkMode.setObjectName("Dark Mode")
        self.darkMode.setCheckable(True)
        self.menuSettings.addAction(self.darkMode)

        self.menuSettings.triggered[QAction].connect(self.changeTheme)

        #Add Menu Language to Settings Menu
        self.menuLanguage = QtWidgets.QMenu(self.menuSettings)
        self.menuLanguage.setObjectName("menuLanguage")
        self.menuSettings.addAction(self.menuLanguage.menuAction())
        ## Add English Option to Menu Language (Settings Menu)
        self.actionEnglish = QtWidgets.QAction(MainWindow)
        self.actionEnglish.setObjectName("actionEnglish")
        self.actionEnglish.setCheckable(True)
        self.menuLanguage.addAction(self.actionEnglish)
        ## Add French Option to Menu Language (Settings Menu)
        self.actionFrancais = QtWidgets.QAction(MainWindow)
        self.actionFrancais.setObjectName("actionFrancais")
        self.actionFrancais.setCheckable(True)
        self.menuLanguage.addAction(self.actionFrancais)

        self.menuLanguage.triggered[QAction].connect(self.changeLanguage)

        #Add Antispoofing Mode to Setting Menu
        self.anti_spoofing = QtWidgets.QAction(MainWindow)
        self.anti_spoofing.setObjectName("antispoofing")
        self.anti_spoofing.setCheckable(True)
        self.menuSettings.addAction(self.anti_spoofing)

        #Add Generate Logs Menu to Setting Menu
        self.logs = QtWidgets.QMenu(self.menuSettings)
        self.logs.setObjectName("logs")
        self.menuSettings.addAction(self.logs.menuAction())
        ## Add PNG Option to Generate Logs Menu (Settings Menu)
        self.logs_png = QtWidgets.QAction(MainWindow)
        self.logs_png.setCheckable(True)
        self.logs.addAction(self.logs_png)
        ## Add CSV Option to Generate Logs Menu (Settings Menu)
        self.logs_csv = QtWidgets.QAction(MainWindow)
        self.logs_csv.setCheckable(True)
        self.logs.addAction(self.logs_csv)
        
        self.logs.triggered[QAction].connect(self.logsGenerator)


        #self.open_csv.clicked.connect(self.Openfile)
        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.capture = cv2.VideoCapture(0)
        #delete_personel("vgg16", "Djamel Hemch")
        #reinteger_all("vgg16")


    """
    METHODS
    """
    def openDashboard(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Dashboard_Ui()
        self.ui.setupUi(self.window)
        self.window.show()

    def openEnrolementAddNew(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_enrolement_addNew()
        self.ui.setupUi(self.window)
        self.window.show()

    def openEnrolementChangeAccess(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_enrolement_ChangeAccess()
        self.ui.setupUi(self.window)
        self.window.show()

    def enrollment_windows(self):
        if self.actionAjouter_personne.isChecked():
            self.removeAccess.setChecked(False)
            self.openEnrolementAddNew()
            self.actionAjouter_personne.setChecked(False)
            

        if self.removeAccess.isChecked():
            self.actionAjouter_personne.setChecked(False)
            self.openEnrolementChangeAccess()
            self.removeAccess.setChecked(False)
            


    def viewCam(self, test):
        _translate = QtCore.QCoreApplication.translate
        
        self.recordBTN.setIcon(QIcon("./imgs/stop.png"))
        self.recordBTN.setToolTip("Stop Face Recognition")
        self.recordBTN.setText(_translate("MainWindow", " Stop Capture"))

        self.EYE_AR_THRESH = 0.15
        self.COUNTER = 0
        frame_rate = 10
        prev = 0
        
        
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        while True:
            stime = time.time()
            # Capture frame-by-frame
            time_elapsed = time.time() - prev
            ret, frame = self.capture.read()
            if time_elapsed > 1. / frame_rate:
                prev = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            height, width, channel = image.shape
            step = channel * width

            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

            # Draw a rectangle around the faces
            for rect in rects:
                x,y,w,h= convert_and_trim_bb(gray, rect) 
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    time.sleep(0.099)


                self.blink_count.setText("Blinks: {}".format(self.COUNTER))
                

            if self.anti_spoofing.isChecked():
                if self.COUNTER >= 5:
                    self.take_pic.setEnabled(True)
                    self.EAR_label.setText("ANTI-SPOOFING PASSED!")
                    self.EAR_label.setStyleSheet("color: green;")
                else:
                    self.take_pic.setEnabled(False)
                    self.EAR_label.setText("WAITING FOR ANTI-SPOOFING")
                    self.EAR_label.setStyleSheet("""color: red;""")
            else :
                #print("anti spoofing disabled")
                self.take_pic.setEnabled(True)
                self.EAR_label.setText("ANTI-SPOOFING DISABLED!")
                self.EAR_label.setStyleSheet("""color: white;""")
            self.cam_label.setPixmap(QPixmap.fromImage(qImg))


            self.show_fps.setText('FPS {:.1f}'.format(1 / (time.time() - stime)))

            if test == 1:
                break
                self.capture.release()
                self.cam_label.clear()
                print("Stopping capture...")

            if cv2.waitKey(0) & 0xFF == ord('z'):
                break



    def capture_image(self):
        
        ret, frame = self.capture.read()
        N = 5
        addon = ''.join(random.choices(string.ascii_uppercase +  string.digits, k = N))
        image_list = []
        self.COUNTER = 0
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                rects = detector(gray, 0)
                for rect in rects:
                    x,y,w,h= convert_and_trim_bb(gray, rect)
                    roi_color = frame[y:y + h, x:x + w]
                    cv2.imwrite(GALLERY_IMAGES_PATH + "frame-{}.jpg".format(addon)  , roi_color)

            except:
                pass
            ts = 0
            found = None

            os.getcwd()
            #Fething the face images captured
            for file_name in glob(GALLERY_IMAGES_PATH+"*.jpg"):
                fts = os.path.getmtime(file_name)
                print(fts)

                if fts > ts:
                    ts = fts
                    found = file_name
                    #Getting the latest captured face image and prints it

            x_train = np.load(GALLERY_PATH+"vgg16_x_train.npy")
            y_train = np.load(GALLERY_PATH+"vgg16_y_train.npy")
            last_image = found

            roi_color = cv2.resize(roi_color, (224, 224),interpolation= cv2.INTER_AREA)  # load an image and resize it to 224,224 like vgg face input size
            roi_color = img_to_array(roi_color)  # convert the image to an array
            roi_color = np.expand_dims(roi_color,axis=0)  # add the 4th dimension as a tensor to inject through the vgg face network
            roi_color = utils.preprocess_input(roi_color, version= 1)  # preprocess the image 1 = vggface resnet = 2)
            feature_vector = vgg_face_model.predict(roi_color)  # extract the features
            face_prediction = prediction_cosine_similarity2(x_train, y_train, feature_vector, 5)[0]

            authorized = np.load(HISTORY_PATH+"authorized.npy")
            access_history = np.load(HISTORY_PATH+"access_history.npy")
            accesstime_history= np.load(HISTORY_PATH+"accesstime_history.npy")
            class_history = np.load(HISTORY_PATH+"class_access_history.npy")
            date_access = np.load(HISTORY_PATH+"date_access.npy")
            time_access = np.load(HISTORY_PATH+"time_access.npy")

            timing = datetime.now()
            starting_worktime= datetime(timing.year, timing.month, timing.day, 8, 0, 0)
            ending_worktime = datetime(timing.year, timing.month, timing.day, 18, 0, 0)
            


            if face_prediction == 'Not Recognized':

                access_history = np.append(access_history, "Rejected" + '\n'+ "(Unrecognized)")
                class_history = np.append(class_history, face_prediction)
                date_access = np.append(date_access, str(timing.year) + "-" + str(timing.month) + "-" + str(timing.day))
                time_access = np.append(time_access,str(timing.hour) + ':' + str(timing.minute) + ':' + str(timing.second))
                self.history.setText(str(face_prediction)+" personnel")

            elif face_prediction in authorized:
                #grant
                if timing > starting_worktime and timing < ending_worktime:
                    accesstime_history = np.append(accesstime_history,"Authorized at"+'\n'+"working hours")
                else:
                    accesstime_history = np.append(accesstime_history,"Authorized after"+'\n' +"working hours")


                access_history = np.append(access_history, "Authorized")
                class_history = np.append(class_history, face_prediction)
                date_access = np.append(date_access, str(timing.year) +"-" + str(timing.month) +"-"+str(timing.day))
                time_access = np.append(time_access, str(timing.hour) +':' + str(timing.minute) +":"+str(timing.second))

                print("Authorized!")
                self.history.setText(str(face_prediction) + ": Is Authorized")
                self.history.setStyleSheet("color: green;""border: 1px solid white;""font-style:bold;")
            else:
                #deny
                if timing > starting_worktime and timing < ending_worktime:
                    accesstime_history = np.append(accesstime_history, "Rejected at" + '\n' + "working hours")
                else:
                    accesstime_history = np.append(accesstime_history, "Rejected after" + '\n' + "working hours")
                
                access_history=np.append(access_history, "Rejected" + '\n'+ "(Recognized)")
                class_history = np.append(class_history, face_prediction)
                date_access = np.append(date_access, str(timing.year) + "-" + str(timing.month) + "-" + str(timing.day))
                time_access = np.append(time_access, str(timing.hour) + ':' + str(timing.minute) + ':' + str(timing.second))
                self.history.setText(str(face_prediction) + ": Is Unauthorized")
                self.history.setStyleSheet("color: red;""border: 1px solid white;"
                                     "font-style:bold;")
                
            np.save(HISTORY_PATH+"access_history.npy",access_history)
            np.save(HISTORY_PATH+"accessTime_history.npy",accesstime_history)
            np.save(HISTORY_PATH+"class_access_history.npy", class_history)
            np.save(HISTORY_PATH+"date_access.npy", date_access)
            np.save(HISTORY_PATH+"time_access.npy", time_access)
            
            input_img = cv2.imread(str(last_image))
            height, width, channel = input_img.shape
            step = channel * width
            qImg = QImage(input_img.data, width, height, step, QImage.Format_RGB888)

            self.face_frame.setPixmap(QPixmap.fromImage(qImg))
            self.history.setAlignment(QtCore.Qt.AlignLeft)
        self.viewCam(0)

    def logsGenerator(self):

        if self.logs_csv.isChecked():
            print("Generating log file in CSV format ....")
            generate_report(GALLERY_PATH, NPY_FILES, PD_COLUMNS, 'csv', REPORT_OUTPUT_PATH)
            self.logs_png.setChecked(False)
        if self.logs_png.isChecked():
            print("Generating log file in PNG format ....")
            generate_report(GALLERY_PATH, NPY_FILES, PD_COLUMNS, 'png', REPORT_OUTPUT_PATH)
            self.logs_csv.setChecked(False)

    def changeTheme(self):
        if self.darkMode.isChecked():
            self.darkMode.setText('Disable Light Mode')
            # setting background color to Light
            MainWindow.setStyleSheet("background-color: #cccccc;"
        "color: #1b1553;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")
            self.menubar.setStyleSheet("""
        QMenuBar
        {
            background-color: #0c0c13 ;
            color: #fff;
        }
        QMenuBar::item
        {
            font-style: Play;
            background-color: #0c0c13;
            color: #fff;
            padding-left : 5px; 
        }
        QMenuBar::item::selected
        {  
            background-color: #1b1553;
            color: #fff;
            
        }
        QMenu
        {   
            background-color: #1e0a30;
            color: #fff;
            padding-left : 5px; 
        }
        QMenu::item::selected
        {
            background-color: #1b1553;
            color: #fff;
            border : 1px solid #ff4b3c;
        }

        """)
            self.recordBTN.setStyleSheet(
                "QPushButton::hover"
                "{"
                "background-color: #1b1553;"
                "color: #eeeeee;"
                "}"
                "border: 1px solid #1b1553;"
            )
            self.take_pic.setStyleSheet(
                "QPushButton::hover"
                "{"
                "background-color: #1b1553;"
                "color: #eeeeee;"
                "}"
                "border: 1px solid #1b1553;"
            )
            self.cam_label.setStyleSheet("border: 1px solid black;"
                                         "font-style:bold;"
                                         )
            self.history.setStyleSheet("border: 1px solid black;"
                                          "font-style:bold;"
                                          )
            self.face_frame.setStyleSheet("border: 1px solid black;"
                                          "font-style:bold;"
                                          )
            # if it is unchecked
        else:
            # set background color back to dARK
            
            self.darkMode.setText('Enable Light Mode')
            MainWindow.setStyleSheet("background-color: #1b1553;"
        "color: #ff4b3c;"
        "font-family: Play;"
        "font-size: 18px;"
        "border-radius: 5px;")

            self.menubar.setStyleSheet("""
        QMenuBar
        {
            background-color: #0c0c13 ;
            color: #fff;
        }
        QMenuBar::item
        {
            font-style: Play;
            background-color: #0c0c13;
            color: #fff;
            padding-left : 5px; 
        }
        QMenuBar::item::selected
        {  
            background-color: #1b1553;
            color: #fff;
            
        }
        QMenu
        {   
            background-color: #1e0a30;
            color: #fff;
            padding-left : 5px; 
        }
        QMenu::item::selected
        {
            background-color: #1b1553;
            color: #fff;
            border : 1px solid #ff4b3c;
        }

        """)

            self.recordBTN.setStyleSheet(
                "QPushButton::hover"
                "{"
                "background-color:rgb(255,255,255);"
                "color: black;"
                "}"
                "border: 1px solid white;"
            )

            self.take_pic.setStyleSheet(
                "QPushButton::hover"
                "{"
                "background-color:rgb(255,255,255);"
                "color: black;"
                "}"
                "border: 1px solid white;"
            )
            self.cam_label.setStyleSheet("border: 1px solid white;"
                                     "font-style:bold;"
                                     )
            self.history.setStyleSheet("border: 1px solid white;"
                                       "font-style:bold;"
                                       )
            self.face_frame.setStyleSheet("border: 1px solid white;"
                                       "font-style:bold;"
                                       )

    def stopVideo(self):
        self.viewCam(1)
        print("capture stopped")

    def changeLanguage(self):
        if self.actionFrancais.isChecked():
            print("francais")
            self.actionEnglish.setChecked(False)
        if self.actionEnglish.isChecked():
            print("english")
            self.actionFrancais.setChecked(False)
        #--------------------Methods -END - --------------------#
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", " FRekoAccess"))
        self.recordBTN.setText(_translate("MainWindow", "Open Camera"))
        self.cam_label.setText(_translate("MainWindow", "Camera Capture Frame"))
        self.take_pic.setText(_translate("MainWindow", "Authenticate"))
        self.face_frame.setText(_translate("MainWindow", "Captured image of face"))
        #self.nom.setText(_translate("MainWindow", "Nom"))
        #self.prenom.setText(_translate("MainWindow", "Prenom"))
        self.history.setText(_translate("MainWindow", "Access Authorizations"))
        #self.subject_info.setText(_translate("MainWindow", " SUBJECT INFO"))
        #self.open_csv.setText(_translate("MainWindow", "Open CSV File"))
        self.menuEnrolement.setTitle(_translate("MainWindow", "Enrollment"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.menuFace_recognition.setTitle(_translate("MainWindow", "Face recognition"))
        self.menuDashboard.setTitle(_translate("MainWindow", "Dashboard"))
        self.actionAjouter_personne.setText(_translate("MainWindow", "New personnel"))
        self.dashboardWindow.setText(_translate("MainWindow", "Open Dashboard"))
        self.show_fps.setText(_translate("MainWindow", ""))
        self.darkMode.setText(_translate("MainWindow", "Enable Light Mode"))
        self.menuLanguage.setTitle(_translate("MainWindow", "Language"))
        self.actionEnglish.setText(_translate("MainWindow", "English"))
        self.actionFrancais.setText(_translate("MainWindow", "Français"))
        self.anti_spoofing.setText(_translate("MainWindow", "Anti-Spoofing"))
        self.EAR_label.setText(_translate("MainWindow", ""))
        self.blink_count.setText(_translate("MainWindow", ""))
        self.logs.setTitle(_translate("MainWindow", "Generate Logs"))
        self.logs_png.setText(_translate("MainWindow", "PNG format"))
        self.logs_csv.setText(_translate("MainWindow", "CSV format"))
        self.removeAccess.setText(_translate("MainWindow", "Change access"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
