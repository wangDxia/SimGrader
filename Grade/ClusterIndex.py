from model import MLPGrader
import pandas as pd
from process_data import read_label, normal_data, newnormal_data, nonormal_data
import cluster.index as indexs
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics, svm
from sklearn.decomposition import PCA
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def cla_problem(index, y_pred, y_test):
    y_pred_or = []
    y_test_or = []
    acc = []
    for i in range(len(index)):
        y_pred_or.append(y_pred[index.index(i)])
        y_test_or.append(y_test[index.index(i)])

    for i in range(15):
        ov_acc = metrics.accuracy_score(
            y_test_or[i * 30 : (i + 1) * 30 - 1], y_pred_or[i * 30 : (i + 1) * 30 - 1]
        )
        acc.append(ov_acc)
    return acc

def cal_time(data):
    for i in range(len(data)):
        if data["test_cases"][i] == 0:
            data["min_memory"][i] = 1000000
            data["max_memory"][i] = 1000000
            data["avg_memory"][i] = 1000000
        if data["test_cases"][i] == 0:
            data["min_runtime"][i] = 2000
            data["max_runtime"][i] = 2000
            data["avg_runtime"][i] = 2000
    return data

if __name__ == "__main__":
    path = "data/grade_feature_similarity_all_vector_ASTNN_nopre.pkl"
    labelpath = "data/codegrade-new.csv"
    indpath = "data/grade_ind_new.pkl"
    ind = pd.read_pickle(indpath)
    data = pd.read_pickle(path)
    label = [int(i / 30) for i in range(450)]
    data = read_label(data, labelpath)
    data = cal_time(data)
    feature, label, index = nonormal_data(data)
    num = 15
    modelstr = "kmeans"
    if modelstr == "kmeans":
        estimator = KMeans(n_clusters=num, random_state=777)
        estimator.fit(feature)
        s = metrics.davies_bouldin_score(feature, estimator.labels_)
        indexs.cal_sihouette(feature, estimator.labels_)
        wae = indexs.WAE(estimator.labels_, np.array(label))
        print(s)
        print(wae)
