import pandas as pd
from process_data import read_label, normal_data, for_count
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import scipy
from sklearn.preprocessing import StandardScaler


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def good_sets(data):

    pro_list = list(set(data["problem_id"]))
    good_set = {}

    for i in range(len(data)):
        if data["alllabel"][i] == 4:
            if data["problem_id"][i] in good_set.keys():
                if len(good_set[data["problem_id"][i]]) <= 3:
                    good_set[data["problem_id"][i]] += [i]
            else:

                good_set[data["problem_id"][i]] = []
                good_set[data["problem_id"][i]] += [i]

    return good_set


def cal_similarity(data, alldata, good_set):
    max_s = -1
    for i in good_set:
        s = cosine_similarity([data], [alldata[i]])
        if max_s <= s:
            max_s = s
    return max_s


def normal(s):
    if s > 0:
        return (s + 1) / 2
    else:
        return abs(s) / 2


def grade(data, good_set):

    style_feature = [
        "IdentifierCount",
        "complexity",
        "unused_vars",
        "unused_lines",
        "code_length",
        "code_vocabulary",
        "code_volume",
        "code_difficulty",
        "code_effort",
        "coding_time",
        "max_runtime",
        "min_runtime",
        "avg_runtime",
        "max_memory",
        "min_memory",
        "avg_memory",
        "test_cases",
        "comment_count",
        "blank_count",
    ]
    sematic_feature = ["simivector"]
    code_result = []
    main_fea = []
    style_fea = []
    smematic_fea = []
    for j in range(len(data)):

        f = []
        code_result.append(data["code_result"][j])
        for i in featureListMain:
            f.append(data[i][j])
        main_fea.append(f)
        f = []
        for i in style_feature:
            f.append(data[i][j])
        style_fea.append(f)
        f = []
        for i in sematic_feature:
            f.append(data[i][j][0])
        smematic_fea.append(f)

    transfer = StandardScaler()
    smematic_fea = transfer.fit_transform(smematic_fea)
    style_fea = transfer.fit_transform(style_fea)
    allsimi = []
    for i in range(len(style_fea)):
        if code_result[i] == 0:
            s = cal_similarity(style_fea[i], style_fea, good_set[data["problem_id"][i]])
            s2 = cal_similarity(
                smematic_fea[i], smematic_fea, good_set[data["problem_id"][i]]
            )
            allsimi.append((normal(s[0][0]) * 50) + (normal(s2[0][0]) * 50))

        else:
            s = cal_similarity(style_fea[i], style_fea, good_set[data["problem_id"][i]])
            s2 = cal_similarity(
                smematic_fea[i], smematic_fea, good_set[data["problem_id"][i]]
            )

            allsimi.append((normal(s[0][0]) * 50) + (normal(s2[0][0]) * 50))

    return allsimi


def max_score(data, ind, good_set, s, minn):

    score = 0
    for i in good_set:
        if data[ind] >= data[i] or data[i] < minn:
            return 1 * s
        elif data[ind] == minn:
            return 0
        else:
            s = (data[ind] - minn) / (data[i] - minn) * s

        if s > score:
            score = s
    return score


def min_score(data, ind, good_set, s, maxn):

    score = 0
    for i in good_set:
        if data[ind] <= data[i] or data[i] > maxn:
            return 1 * s
        elif data[ind] == maxn:
            return 0
        else:
            s = (maxn - data[ind]) / (maxn - data[i]) * s

        if s > score:
            score = s
    return score


def find_code(score, data):
    correct = []
    incorrect = []

    for i in range(len(data["code_result"])):
        if data["code_result"][i] == 0:
            correct.append(i)
        else:
            incorrect.append(i)
    cor_s = []
    for i in range(len(correct)):
        cor_s.append(score[correct[i]])
    max_cor = correct[cor_s.index(max(cor_s))]
    min_cor = correct[cor_s.index(min(cor_s))]
    incor_s = []
    for i in range(len(incorrect)):
        incor_s.append(score[incorrect[i]])
    max_incor = incorrect[incor_s.index(max(incor_s))]
    min_incor = incorrect[incor_s.index(min(incor_s))]

    return [max_cor, min_cor, max_incor, min_incor]

if __name__ == "__main__":
    path = "data/grade_feature_similarity_all_vector_ASTNN_nopre.pkl"
    labelpath = "data/codegrade-new.csv"
    indpath = "data/grade_ind_new.pkl"
    alldata = pd.read_pickle("data/feature_grade0331.pkl")
    alldata = alldata.rename(columns={"probelm_id": "problem_id"})
    data = pd.read_pickle("data/grade.pkl")
    data = cal_time(data)
    good_set = good_sets(data)
    final_score = grade_new(data, good_set)
    compare = {"manul": data["alllabel"], "score": final_score}
    score = pd.DataFrame(compare)
