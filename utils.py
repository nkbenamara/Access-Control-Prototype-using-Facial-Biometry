from statistics import mode
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import os
import pandas as pd
import dataframe_image as dfi
from datetime import datetime

from os.path import exists
from paths import *

def convert_and_trim_bb(image, rect):
	'''
    This function convert dlib rectangle to x,y,h,w coordinates
    '''
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def eye_aspect_ratio(eye):
    '''
    This function computes the eye aspect ratio for the eye-blink anti-spoofing technique
    '''
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A + B) / (2.0 * C)
    
    return ear

def findCosineDistance(source_representation, test_representation):
    '''
    This function calculates the  Cosine Distance between two feature vectors
    '''
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def prediction_cosine_similarity2(x_train, y_train, x_test, neighbors):
    '''
    This function classifies a new face based on the saved NPY files (x_train, y_train)
    The neighbor input defines the n closest feature vector for a k-NN classification
    '''
    size = np.shape(x_train)[0]  
    distance = []
    sorted_distance = []
    predictions = []
    y_pred = 0

    for i in range(0, size):  
        d = findCosineDistance(x_train[i], x_test[0])
        distance.append(d)
    sorted_distance = np.sort(distance)

    if sorted_distance[0] >= 0.2:
        y_pred = np.array(["Not Recognized"])

    else:
        for j in range(0, neighbors):
            indice = np.where(distance == sorted_distance[j])[0][0]
            predictions.append(y_train[indice])
            try:
                y_pred = mode(predictions)
            except:
                indice_nomode = np.where(distance == sorted_distance[0])[0][0]
                y_pred = y_train[indice_nomode]
    return y_pred

def donutGenerator(path, output_filename):
    '''
    This function generates a donuts figures to diplay it in the dashboard view 
    '''
    auth = np.load(path)

    data = np.unique(auth, return_counts=True)[1]
    recipe = np.unique(auth, return_counts=True)[0]

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
    plt.rcParams['font.size'] = 9.0

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),horizontalalignment=horizontalalignment, **kw)
    fig = plt.gcf()
    fig.set_size_inches(4, 2)
    plt.savefig(output_filename, transparent = True)

def generate_report(source_path, data_files, columns_titles, output_type, output_path):
    '''
    This function generates a report from several npy files, used to store access data
    The available output extensions are: CSV and PNG
    '''
    data = np.array([np.load(source_path + data_files[0])])
    accepted_output_extensions = ["csv", "png"]
    for data_file in data_files[1:]:
        data = np.concatenate((data, np.array([np.load(source_path + data_file)])), axis=0)
    dataDf = pd.DataFrame(data=np.transpose(data), columns=columns_titles)

    os.makedirs(output_path, exist_ok=True)

    output_file_name = 'report' + datetime.now().strftime('%Y_%m_%d-at-%H_%M_%S.')
    if output_type == 'csv':
        dataDf.to_csv(output_path + output_file_name + output_type)
        print('The report is generated in a {} format and saved at {}{}{}'.format(output_type.upper(), output_path,
                                                                                  output_file_name, output_type))

    elif output_type == 'png':
        dataDf_png = dataDf.style.background_gradient()
        dfi.export(dataDf_png, output_path + 'report' + datetime.now().strftime('%Y_%m_%d-at-%H_%M_%S.' + output_type))
        print('The report is generated in a {} format and saved at {}{}{}'.format(output_type.upper(), output_path,
                                                                                  output_file_name, output_type))

    else:
        print('The selected output extension is invalid. You can choose the following extensions {}'.format(
            accepted_output_extensions))

    return dataDf


def delete_personel(class_name):
    '''
    This function deletes a selected personel from the gallery (x_train and y_train)
    '''
    
    try: 
        x_train=np.load(GALLERY_PATH+'x_train.npy')
        y_train=np.load(GALLERY_PATH+'y_train.npy')
    except :
        print('This model is not recognized.')
        return None
    
    if class_name in y_train:
        indexes = np.where(y_train ==class_name)[0]
    else:
        print('{} does not exist in the gallery'.format(class_name))
        return None

    if exists(GALLERY_PATH+'deleted_y_train.npy'):
        deleted_x_train=np.load(GALLERY_PATH+'deleted_x_train.npy')
        deleted_y_train=np.load(GALLERY_PATH+'deleted_y_train.npy')

        deleted_x_train = np.concatenate((deleted_x_train,np.expand_dims(x_train[indexes[0]], axis=0)), axis=0)
        deleted_y_train = np.concatenate((deleted_y_train, np.expand_dims(y_train[indexes[0]], axis=0)), axis=0)
    else:
        deleted_x_train= np.array([])
        deleted_y_train= np.array([])

        deleted_x_train= np.expand_dims(np.append(deleted_x_train, x_train[indexes[0]]), axis=0)
        deleted_y_train = np.expand_dims(np.append(deleted_y_train, y_train[indexes[0]]), axis=0)


    for index in indexes[1:] :
        feature_to_delete=  np.expand_dims(x_train[index], axis=0)
        label_to_delete = np.expand_dims(y_train[index], axis=0)
        deleted_x_train = np.concatenate((deleted_x_train,feature_to_delete), axis=0)
        deleted_y_train = np.concatenate((deleted_y_train, label_to_delete), axis=0)
    
    x_train = np.delete(x_train, indexes, axis=0)
    y_train = np.delete(y_train, indexes, axis=0)

    np.save(GALLERY_PATH+'deleted_x_train.npy', deleted_x_train)
    np.save(GALLERY_PATH+'deleted_y_train.npy', deleted_y_train)

    np.save(GALLERY_PATH+'x_train.npy', x_train)
    np.save(GALLERY_PATH+'y_train.npy', y_train)

    print ("The {} samples have been deleted with success".format(class_name))

    return None
    
def reinteger_all():
    '''
    This function reintegrates all removed collaborators again to the gallery
    '''
    
    try: 
        if not exists(GALLERY_PATH+'deleted_y_train.npy'):
            return print('Any collaborator has been removed previously from the gallery to reintegrate')
        else:
            deleted_x_train=np.load(GALLERY_PATH+'deleted_x_train.npy')
            deleted_y_train=np.load(GALLERY_PATH+'deleted_y_train.npy')
            if deleted_y_train.size==0:
                return print('Any collaborator has been removed previously from the gallery to reintegrate')
    except :
        print('This model is not recognized.')
        return None

    x_train=np.load(GALLERY_PATH+'x_train.npy')
    y_train=np.load(GALLERY_PATH+'y_train.npy')

    x_train=np.concatenate((x_train, deleted_x_train), axis=0)
    y_train=np.concatenate((y_train, deleted_y_train), axis=0)

    

    np.save(GALLERY_PATH+'x_train.npy', x_train)
    np.save(GALLERY_PATH+'y_train.npy', y_train)
    
    print('==================================================================================')
    print('All removed collaborators have been reintegrated to the gallery again with sucess')
    print('The reintegrated collaborators are :{}'.format(np.unique(deleted_y_train, return_counts=True)[0]))
    print('==================================================================================')

    deleted_x_train=np.array([])
    deleted_y_train=np.array([])
    np.save(GALLERY_PATH+'deleted_x_train.npy', deleted_x_train)
    np.save(GALLERY_PATH+'deleted_y_train.npy', deleted_y_train)
    
    return None

def delete_personel(class_name):
    '''
    This function deletes a selected personel from the gallery (x_train and y_train)
    '''
    
    try:
        x_train=np.load(GALLERY_PATH+'x_train.npy')
        y_train=np.load(GALLERY_PATH+'y_train.npy')
    except:
        print('This model is not recognized.')
        return None
    
    if class_name in y_train:
        indexes = np.where(y_train ==class_name)[0]
    else:
        print('{} does not exist in the gallery'.format(class_name))
        return None

    if exists(GALLERY_PATH+'deleted_y_train.npy'):
        deleted_x_train=np.load(GALLERY_PATH+'deleted_x_train.npy')
        deleted_y_train=np.load(GALLERY_PATH+'deleted_y_train.npy')

        deleted_x_train = np.concatenate((deleted_x_train,np.expand_dims(x_train[indexes[0]], axis=0)), axis=0)
        deleted_y_train = np.concatenate((deleted_y_train, np.expand_dims(y_train[indexes[0]], axis=0)), axis=0)
    else:
        deleted_x_train= np.array([])
        deleted_y_train= np.array([])

        deleted_x_train= np.expand_dims(np.append(deleted_x_train, x_train[indexes[0]]), axis=0)
        deleted_y_train = np.expand_dims(np.append(deleted_y_train, y_train[indexes[0]]), axis=0)


    for index in indexes[1:] :
        feature_to_delete=  np.expand_dims(x_train[index], axis=0)
        label_to_delete = np.expand_dims(y_train[index], axis=0)
        deleted_x_train = np.concatenate((deleted_x_train,feature_to_delete), axis=0)
        deleted_y_train = np.concatenate((deleted_y_train, label_to_delete), axis=0)
    
    x_train = np.delete(x_train, indexes, axis=0)
    y_train = np.delete(y_train, indexes, axis=0)

    np.save(GALLERY_PATH+'deleted_x_train.npy', deleted_x_train)
    np.save(GALLERY_PATH+'deleted_y_train.npy', deleted_y_train)

    np.save(GALLERY_PATH+'x_train.npy', x_train)
    np.save(GALLERY_PATH+'y_train.npy', y_train)

    print ("The {} samples have been deleted with success".format(class_name))

    return None

def load_actual_collaborators():
    '''
    This function shows all available collaborators in the gallery
    '''
    try: 
        y_train=np.load(GALLERY_PATH+'y_train_camera1.npy')
    except:
        print('This model is not recognized.')
        return None
    y_train = np.unique(y_train, return_counts=True) 
    return y_train[0]