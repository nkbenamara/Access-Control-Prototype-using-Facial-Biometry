from statistics import mode

import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from glob import glob
import os

import pandas as pd
import dataframe_image as dfi
from datetime import datetime
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def prediction_cosine_similarity2(x_train, y_train, x_test, neigbors):
    size = np.shape(x_train)[0]  # nombre de vecteurs caracterestique dans le x_train
    distance = []
    sorted_distance = []
    predictions = []
    y_pred = 0
    for i in range(0, size):  # calcul distance entre x_train et le vecteur a classifier
        d = findCosineDistance(x_train[i], x_test[0])
        distance.append(d)
    sorted_distance = np.sort(distance)
    print(sorted_distance[0])
    if sorted_distance[0] >= 0.2:
        y_pred = np.array(["Not Recognized"])
        print(y_pred)
    else:
        for j in range(0, neigbors):
            indice = np.where(distance == sorted_distance[j])[0][0]
            predictions.append(y_train[indice])
            try:
                y_pred = mode(predictions)
            except:
                indice_nomode = np.where(distance == sorted_distance[0])[0][0]
                y_pred = y_train[indice_nomode]
    print(y_pred)
    return y_pred








def donutGenerator(path, output_filename):
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