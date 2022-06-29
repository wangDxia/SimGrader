import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from similarity_model.ASTNN.model import BatchProgramClassifier
from pycparser import c_parser
from similarity_model.ASTNN.prepare_data import get_blocks as func


def tree_to_index(node, vocab, max_token):
    token = node.token
    result = [vocab.index(token) if token in vocab else max_token]
    children = node.children
    for child in children:
        result.append(tree_to_index(child, vocab, max_token))
    return result


def trans2seq(r, vocab, max_token):
    blocks = []
    func(r, blocks)
    tree = []
    for b in blocks:
        btree = tree_to_index(b, vocab, max_token)
        tree.append(btree)
    return tree


def general_astnndata(codestr, word2vec):

    vocab = list(word2vec.key_to_index.keys())
    max_token = len(word2vec.key_to_index)
    parser = c_parser.CParser()
    ast = parser.parse(codestr)
    tree = trans2seq(ast, vocab, max_token)
    return tree


def similarity_class(model_dic, codestr1, codestr2, word2vec):

    embeddings = np.zeros(
        (word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32"
    )
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 5
    OUT_DIM = 64
    BATCH_SIZE = 1
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
        False,
    )

    model.load_state_dict(model_dic)
    test_inputs = general_astnndata(codestr1, word2vec)
    test_inputs2 = general_astnndata(codestr2, word2vec)
    output = model([test_inputs], [test_inputs2])
    return output
