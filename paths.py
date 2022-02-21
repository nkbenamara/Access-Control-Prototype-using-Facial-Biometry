import os
import numpy as np


MAIN_PATH = os.getcwd()

GALLERY_PATH = os.getcwd() + '/gallery/'
GALLERY_IMAGES_PATH= os.getcwd() + '/gallery_images/'
REPORT_OUTPUT_PATH = os.getcwd() + '/report_logs/'
HISTORY_PATH= os.getcwd() + '/history/'
STAT_PATH= os.getcwd() + '/stats/'

FACE_DETECTION_MODELS = os.getcwd()+'/pretrained_models/face_detection/'
EYE_DETECTION_MODELS = os.getcwd()+'/pretrained_models/eye_detection/'

NPY_FILES = ['class_access_history.npy', 'access_history.npy', 'date_access.npy', 'time_access.npy',
             'accessTime_history.npy']
PD_COLUMNS = ['Subject', 'Status', 'Date', 'At', 'Period']


