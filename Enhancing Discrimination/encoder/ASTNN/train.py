import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import loss as losses


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx : idx + bs]
    code1, code2, labels = [], [], []
    for _, item in tmp.iterrows():
        code1.append(item[0])
        code2.append(item[1])
        labels.append(item[2])
    return code1, code2, torch.LongTensor(labels)


if __name__ == "__main__":
    root = "data/split/"
    is_pretrain = True
    fix_weight = False
    modelpath = "astnnpretrainInfoNCE"
    train_data = pd.read_pickle(root + "trainblocks.pkl")
    val_data = pd.read_pickle(root + "devblocks.pkl")
    test_data = pd.read_pickle(root + "testblocks.pkl")
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    word2vec = Word2Vec.load(root + "/node_w2v_128").wv
    embeddings = np.zeros(
        (word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32"
    )
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 5
    EPOCHS = 15
    OUT_DIM = 64
    BATCH_SIZE = 64
    USE_GPU = False
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramClassifier(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        MAX_TOKENS + 1,
        ENCODE_DIM,
        OUT_DIM,
        LABELS,
        BATCH_SIZE,
        USE_GPU,
        embeddings,
        fix_weight,
    )
    if is_pretrain:
        pretrained_dict = torch.load("data/model/astnn_pretrainInfoNCE")[
            "model_state_dict"
        ]
        model_dict = model.state_dict()
        pretrained_dict = {
            "emblayer." + k: v
            for k, v in pretrained_dict.items()
            if "emblayer." + k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = losses.FocalLoss(weight=[0.25, 0.25, 0.25, 0.2, 0.2])
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print("Start training...")
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
            if i + BATCH_SIZE > len(train_data):
                break
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            code1_inputs, code2_inputs, train_labels = batch
            if USE_GPU:
                code1_inputs, code2_inputs, train_labels = (
                    code1_inputs,
                    code2_inputs,
                    train_labels.cuda(),
                )

            model.zero_grad()
            model.batch_size = len(train_labels)

            output = model(code1_inputs, code2_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(code1_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for i in tqdm(range(0, len(val_data), BATCH_SIZE)):
            if i + BATCH_SIZE > len(val_data):
                break
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_inputs2, val_labels = batch
            if USE_GPU:
                val_inputs, val_inputs2, val_labels = (
                    val_inputs,
                    val_inputs2,
                    val_labels.cuda(),
                )

            model.batch_size = len(val_labels)

            output = model(
                val_inputs,
                val_inputs2,
            )

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc / total > best_acc:
            best_model = model
        print(
            "[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,"
            " Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s"
            % (
                epoch + 1,
                EPOCHS,
                train_loss_[epoch],
                val_loss_[epoch],
                train_acc_[epoch],
                val_acc_[epoch],
                end_time - start_time,
            )
        )

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    y_true = []
    predicts = []
    model = best_model
    for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
        if i + BATCH_SIZE > len(test_data):
            break
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_inputs2, test_labels = batch
        if USE_GPU:
            test_inputs, test_inputs2, test_labels = (
                test_inputs,
                test_inputs2,
                test_labels.cuda(),
            )

        model.batch_size = len(test_labels)

        output = model(
            test_inputs,
            test_inputs2,
        )

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
        predicts += predicted.tolist()
        y_true += test_labels.tolist()

    print("Testing results(Acc):", total_acc.item() / total)

    t = classification_report(y_true, predicts, target_names=["0", "1", "2", "3", "4"])
    y = [str(i) for i in y_true]
    p = [str(i) for i in predicts]
    m = confusion_matrix(y, p, labels=["0", "1", "2", "3", "4"])

    print("Total testing results(P,R,F1):%.3f" % (total_acc.item() / total))
    print(t)
    print(m)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, modelpath)
