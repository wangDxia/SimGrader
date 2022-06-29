from model import MLPGrader
import pandas as pd
from process_data import read_label, normal_data
from model import MLPGrader
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler  # 标准化函数导入
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_k_fold_data(k, i, X, y):
    transfer = StandardScaler()
    random_state = 1
    X = transfer.fit_transform(X)
    # X = normalization(X)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)

        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid


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


def train_MLP(feature, label):
    input_dim = len(feature[0])
    out_dim = 5
    batch_size = 1
    num_epoch = 80
    all_acc = 0
    y_pred = []
    y_true = []
    for i in range(5):
        model = MLPGrader(input_dim, out_dim, batch_size)
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=0.0005)
        loss_function = torch.nn.CrossEntropyLoss()
        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        train_dataset = TensorDataset(train_x, train_y)  # 转换数据集到数据加载器
        train_iter = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataset = TensorDataset(test_x, test_y)
        test_iter = DataLoader(test_dataset, batch_size=1)
        max_acc = 0
        last_pred = []
        for epoch in range(num_epoch):
            total_acc = 0
            total_loss = 0
            for x, y in train_iter:
                model.zero_grad()
                y_hat = model(x)
                loss = loss_function(y_hat, y)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(y_hat.data, 1)
                total_acc += (predicted == y).sum()
                total_loss += loss.item()
            print("MLP Train ACC:", total_acc / len(train_iter))
            print("MLP Train loss:", total_loss / len(train_iter))
            test_acc = 0

            pred = []
            for t_x, t_y in test_iter:
                y_pre = model(t_x)
                _, predicted = torch.max(y_pre.data, 1)
                test_acc += (predicted == t_y).sum()
                pred.append(predicted)
            if max_acc < test_acc / len(test_iter):
                max_acc = test_acc / len(test_iter)
                last_pred = pred
        y_pred += list(last_pred)
        y_true += list(test_y)
        all_acc += max_acc
    f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    r = metrics.recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    P = metrics.precision_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
    )
    print("F:", f1)
    print("R:", r)
    print("P:", P)

    print("MLP Test ACC:", all_acc / 5)
    return y_pred


def train_bayes(feature, label):
    all_acc = 0
    for i in range(5):
        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        clf = MultinomialNB()
        clf.fit(train_x, train_y)
        y_test_pred = clf.predict(test_x)
        ov_acc = metrics.accuracy_score(y_test_pred, test_y)
        print("bayes result:", ov_acc)
        all_acc += ov_acc
    print("AVE bayes result:", all_acc / 5)
    return y_pred

def train_SVM(feature, label):
    for j in range(1, 2):
        print("C:", j)
        all_acc = 0
        y_pred = []
        y_true = []
        for i in range(5):
            random_state = 1
            train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
            model = svm.SVC(
                kernel="rbf",
                gamma=0.05,
                C=10,
                decision_function_shape="ovo",
                random_state=random_state,
            )
            clt = model.fit(train_x, train_y)
            y_test_pred = clt.predict(test_x)
            ov_acc = metrics.accuracy_score(y_test_pred, test_y)
            print("SVM result:", ov_acc)
            all_acc += ov_acc
            y_pred += list(y_test_pred)
            y_true += list(test_y)
        f1 = metrics.f1_score(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
        )
        r = metrics.recall_score(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
        )
        P = metrics.precision_score(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
        )
        print("F:", f1)
        print("R:", r)
        print("P:", P)
        print("AVE SVM result:", all_acc / 5)
    return y_pred


def train_RandomForest(feature, label):
    all_acc = 0
    y_pred = []
    y_true = []
    for i in range(5):

        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        model = RandomForestClassifier(
            max_depth=10,
            max_features=5,
            min_samples_split=4,
            n_estimators=132,
            criterion="gini",
            random_state=1,
        )

        model.fit(train_x, train_y)
        y_test_pred = model.predict_proba(test_x)
        score1 = model.score(test_x, test_y)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        acc = metrics.accuracy_score(y_test_pred, test_y)
        all_acc += acc
        print("RandomForest result:", acc)
        y_pred += list(y_test_pred)
        y_true += list(test_y)
    f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    r = metrics.recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    P = metrics.precision_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
    )
    print("F:", f1)
    print("R:", r)
    print("P:", P)
    print("RandomForest result:", all_acc / 5)
    return y_pred

def train_DecisionTree(feature, label):
    all_acc = 0
    y_pred = []
    y_true = []
    for i in range(5):
        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            min_samples_leaf=17,
            random_state=2,
            splitter="best",
            max_depth=12,
        )
        # print(clf)
        clf.fit(train_x, train_y)
        y_test_pred = clf.predict(test_x)
        acc = metrics.accuracy_score(y_test_pred, test_y)
        # acc = clf.score(test_x, test_y)

        all_acc += acc
        print("DecisionTree result:", acc)
        y_pred += list(y_test_pred)
        y_true += list(test_y)
    f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    r = metrics.recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    P = metrics.precision_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
    )
    print("F:", f1)
    print("R:", r)
    print("P:", P)

    print("DecisionTree result:", all_acc / 5)
    return y_pred


def train_Lasso(feature, label):
    all_acc = 0

    for i in range(5):
        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        alpha = 0.001
        lasso = Lasso(alpha=alpha)
        lasso.fit(train_x, train_y)
        acc = lasso.score(test_x, test_y)
        all_acc += acc
        print("Lasso result:", acc)
    print("Lasso result:", all_acc / 5)
    return y_pred


def train_GBDT(feature, label):
    all_acc = 0
    y_pred = []
    y_true = []
    for i in range(5):
        train_x, train_y, test_x, test_y = get_k_fold_data(5, i, feature, label)
        gbdt = GradientBoostingClassifier(
            loss="deviance",
            n_estimators=185,
            subsample=0.8,
            min_samples_split=4,
            min_samples_leaf=16,
            max_depth=16,
            random_state=1,
        )
        gbdt.fit(train_x, train_y)
        y_test_pred = gbdt.predict(test_x)
        acc = metrics.accuracy_score(y_test_pred, test_y)
        all_acc += acc
        print("GBDT result:", acc)
        y_pred += list(y_test_pred)
        y_true += list(test_y)
    f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    r = metrics.recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
    P = metrics.precision_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted"
    )
    print("F:", f1)
    print("R:", r)
    print("P:", P)

    print("GBDT result:", all_acc / 5)
    return y_pred


if __name__ == "__main__":
    path = "data/grade_feature_similarity_all_newASTNN.pkl"
    labelpath = "data/codegrade-new.csv"
    indpath = "data/grade_ind_new.pkl"
    ind = pd.read_pickle(indpath)
    data = pd.read_pickle(path)
    data = read_label(data, labelpath)
    feature, label, index = normal_data(data)
    # feature = PCA(n_components=16).fit_transform(feature)
    modelstr = "DecisionTree"
    if modelstr == "MLP":
        y_pred = train_MLP(feature, label)
    elif modelstr == "SVM":
        y_pred = train_SVM(feature, label)
    elif modelstr == "RandomForest":
        y_pred = train_RandomForest(feature, label)

    elif modelstr == "LinearRegression":
        y_pred = train_LinearRegression(feature, label)
    elif modelstr == "DecisionTree":
        y_pred = train_DecisionTree(feature, label)
    else:
        y_pred = train_GBDT(feature, label)
    pro_acc = cla_problem(index, y_pred, label)
    pro_acc = {"acc": pro_acc}
    pro_acc = pd.DataFrame(pro_acc)
    pro_acc.to_csv(modelstr + ".csv")
