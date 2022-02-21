from keras_vggface.vggface import VGGFace
import cv2
from paths import *
import dlib


vgg_face_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
face_cascade = cv2.CascadeClassifier(FACE_DETECTION_MODELS+'haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_DETECTION_MODELS+ "shape_predictor_68_face_landmarks.dat")