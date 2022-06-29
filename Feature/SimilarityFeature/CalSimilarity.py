from similarity_model.TBCNN.tbcnn_similarity import similarity_class as tbcnn_similarity
from similarity_model.ASTNN.astnn_similarity import similarity_class as astnn_similarity
import pandas as pd
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from zss import Node, simple_distance
from pycparser import c_parser, c_ast
import json
from tqdm import tqdm
import torch
import random
from gensim.models.word2vec import Word2Vec


def CalSimilarityClass(codestr, goodset, model_dic, word2vec, model):
    min_class = 4
    for i in goodset:
        if model == "TBCNN":
            out = tbcnn_similarity(model_dic, codestr, i, word2vec)
        if model == "ASTNN":
            out = astnn_similarity(model_dic, codestr, i, word2vec)
        _, predicted = torch.max(out.data, 1)
        out = predicted.tolist()[0]
        if out < min_class:
            min_class = out

    return min_class


def CalSimilarityFeature(data, goodsets, model_dic, word2vec, model):
    similarity_feature = []
    for i in range(len(data)):
        if data["code_result"][i] != 0:
            min_class = CalSimilarityClass(
                data["code_str"][i],
                goodsets[data["problem_id"][i]],
                model_dic,
                word2vec,
                model,
            )
            similarity_feature.append(min_class)
        else:
            similarity_feature.append(0)

    data["similarity_feature"] = similarity_feature
    return data


def ClusterGoodset(featurelist, goodsets):

    import numpy as np

    vec = np.array(featurelist)
    vec = vec
    # clu_pre = DBSCAN(eps=0.001, min_samples=2).fit_predict(vec)
    km = KMeans(n_clusters=15).fit(vec)
    clu_pre = km.labels_
    sets = set(clu_pre.tolist())

    goodset = []
    for i in sets:
        list = [j for j, v in enumerate(clu_pre.tolist()) if v == i]
        goodset.append(goodsets[random.sample(list, 1)[0]])

    return goodset


def Extract_goodset(data, cluster=False):
    prolist = []
    goodset = {}
    list = [
        "for_count",
        "while_count",
        "if_count",
        "code_length",
        "IdentifierCount",
        "code_vocabulary",
    ]
    featurelist = {}
    for i in range(len(data)):
        if data["problem_id"][i] not in goodset.keys() and data["code_result"][i] == 0:
            goodset[data["problem_id"][i]] = []
            featurelist[data["problem_id"][i]] = []
            goodset[data["problem_id"][i]].append(data["code_str"][i])
            v = []
            for f in list:
                v.append(data[f][i])
            featurelist[data["problem_id"][i]].append(v)
        elif data["problem_id"][i] in goodset.keys() and data["code_result"][i] == 0:
            goodset[data["problem_id"][i]].append(data["code_str"][i])
            v = []
            for f in list:
                v.append(data[f][i])
            featurelist[data["problem_id"][i]].append(v)

    if cluster:
        for i in goodset.keys():
            goodset[i] = ClusterGoodset(featurelist[i], goodset[i])

    return goodset


if __name__ == "__main__":

    datapath = "data/feature_grade0331.pkl"
    indpath = "data/grade_ind_new.pkl"
    model = "ASTNN"
    modelpath = "similarity_model/ASTNN/models/astnn-infonce"
    model_dic = torch.load(modelpath)["model_state_dict"]
    data = pd.read_pickle(datapath)
    data = data.rename(columns={"probelm_id": "problem_id"})
    word2vec = Word2Vec.load("data/newnode_w2v_128").wv
    index = pd.read_pickle(indpath)
    allindex = []
    feature = {}
    for i in range(len(pro_list)):
        allindex += index["ind"][i]
    grade_feature = (data.iloc[allindex]).reset_index(drop=True)
    goodsets = Extract_goodset(data, cluster=False)
    grade_data = CalSimilarityFeature(
        grade_feature, goodsets, model_dic, word2vec, model
    )
    grade_data.to_pickle("data/grade_feature_similarity_all_new" + model + ".pkl")
    print("break")
