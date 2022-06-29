import pandas as pd
import os
from Structure_count import get_Structurecount
from utils import feature_list
import random
import numpy as np
from sklearn.decomposition import PCA


def creat_file(codestr, rootpath, filepath):
    filename = os.path.join(rootpath, filepath)
    file = open(filename, "w")
    file.write(codestr)
    file.close()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normal_data(data):

    feature_data = []
    nomal_f = []
    for i in range(len(data)):
        feature = []
        for f in feature_list:

            feature.append(data[f][i])
        nomal_f.append(feature)
        if len(nomal_f) == 30:
            feature_data += normalization(nomal_f).tolist()
            nomal_f = []

    label = data["alllabel"]

    feature_data = np.array(feature_data)
    label = np.array(label)
    index = [i for i in range(len(feature_data))]

    SEED = 1
    random.seed(SEED)
    random.shuffle(index)

    feature_data = feature_data[index]
    label = label[index]
    return feature_data, label, index


def newnormal_data(data):

    feature_data = []
    nomal_f = []
    vects = []
    for i in range(len(data)):
        feature = []
        for f in feature_list:
            if f != "simivector":
                feature.append(data[f][i])
            else:
                vects.append(list(data[f][i]))
        nomal_f.append(feature)
        if len(nomal_f) == 30:
            n = normalization(nomal_f).tolist()
            feature_data += n
            nomal_f = []

    label = data["alllabel"]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    vect = PCA(n_components=18).fit_transform(list(data['simivector']))
    for i in range(len(data)):
        feature_data[i] += vect[i].tolist()


    feature_data = np.array(feature_data)
    label = np.array(label)
    index = [i for i in range(len(feature_data))]

    SEED = 1
    random.seed(SEED)
    random.shuffle(index)

    feature_data = feature_data[index]
    label = label[index]
    return feature_data, label, index


def nonormal_data(data):

    feature_data = []
    nomal_f = []
    vects = []
    label = []
    for i in range(len(data)):
        feature = []
        label.append(int(i / 30))
        for f in feature_list:
            if f != "simivector":
                feature.append(data[f][i])
            else:
                vects.append(list(data[f][i]))
                # feature +=data[f][i]
        nomal_f.append(feature)
        if len(nomal_f) == 30:

            feature_data += nomal_f
            nomal_f = []

    vect = PCA(n_components=18).fit_transform(list(data["simivector"]))
    for i in range(len(data)):
        feature_data[i] += vect[i].tolist()
    feature_data = np.array(feature_data)
    label = np.array(label)
    index = [i for i in range(len(feature_data))]

    SEED = 1
    random.seed(SEED)
    random.shuffle(index)

    feature_data = feature_data[index]
    label = label[index]
    return feature_data, label, index


def read_label(data, labelpath):

    labelcsv = pd.read_csv(labelpath)
    data["alllabel"] = labelcsv["allscore"]
    for_c = []
    if_c = []
    for i in range(len(data)):
        creat_file(data["code_str"][i], "D:/", "code.c")
        fs, _, ifs = get_Structurecount("D:/code.c")
        for_c.append(fs)
        if_c.append(ifs)
    data["for_count"] = for_c
    data["if_count"] = if_c

    return data


def for_count(data):

    for_c = []
    if_c = []
    for i in range(len(data)):
        creat_file(data["code_str"][i], "D:/", "code.c")
        fs, _, ifs = get_Structurecount("D:/code.c")
        for_c.append(fs)
        if_c.append(ifs)
    data["for_count"] = for_c
    data["if_count"] = if_c

    return data


