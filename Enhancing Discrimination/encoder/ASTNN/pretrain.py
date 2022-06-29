import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramPretrain
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import loss
from tqdm import tqdm


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx : idx + bs]
    code1, code2 = [], []
    for _, item in tmp.iterrows():
        code1.append(item["AST1"])
        code2.append(item["AST2"])
    return code1, code2


if __name__ == "__main__":
    root = "data/"
    data = pd.read_pickle(root + "astnnpredata_3.pkl")
    data = data.sample(frac=1).reset_index(drop=True)
    l = "InfoNCEt1"
    modelpath = root + "model/astnn_pretrain" + l
    word2vec = Word2Vec.load(root + "/node_w2v_128").wv
    embeddings = np.zeros(
        (word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32"
    )
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    OUT_DIM = 32
    EPOCHS = 15
    BATCH_SIZE = 32
    USE_GPU = False

    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramPretrain(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        MAX_TOKENS + 1,
        ENCODE_DIM,
        OUT_DIM,
        BATCH_SIZE,
        USE_GPU,
        embeddings,
    )
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    # loss_function = loss.InfoNCELoss(batch_size=BATCH_SIZE, device='cpu',temperature=0.1)
    loss_function = loss.SemiHardTripletLoss(device="cpu")

    train_loss_ = []

    print("Start training...")
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            if i + BATCH_SIZE > len(data):
                break
            batch = get_batch(data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_pos = batch
            if USE_GPU:
                train_inputs, train_pos = train_inputs, train_pos

            model.zero_grad()
            model.batch_size = len(train_inputs)
            model.hidden = model.init_hidden()
            emb1 = model(train_inputs)
            emb2 = model(train_pos)

            loss = loss_function(emb1, emb2)
            loss.backward()
            optimizer.step()

            # calc training acc

            total_loss += loss.item() * len(train_inputs)
            total += len(train_inputs)
        train_loss_.append(total_loss / total)
        end_time = time.time()
        print(
            "[Epoch: %3d/%3d] Training Loss: %.4f,Time Cost: %.3f s"
            % (epoch + 1, EPOCHS, train_loss_[epoch], end_time - start_time)
        )

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, modelpath)
