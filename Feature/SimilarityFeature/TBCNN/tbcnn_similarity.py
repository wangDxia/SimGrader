import pandas as pd
import random
import torch
import pickle
import time
import numpy as np
from similarity_model.TBCNN.model import TBCNNCLassifier
import math


from pycparser import c_parser
import similarity_model.TBCNN.newtrees as Trees


def general_tbcnndata(codestr, word2vec):

    vocab = list(word2vec.key_to_index.keys())
    max_token = len(word2vec.key_to_index)
    parser = c_parser.CParser()
    ast = parser.parse(codestr)
    tree = Trees.parse(ast)
    nodes, children = Trees.gen_onesamples(tree, vocab, max_token, None, None, 1)
    return nodes, children


def truncated_normal_(
    tensor, mean=0, std=0.09
):  # https://zhuanlan.zhihu.com/p/83609874  tf.trunc_normal()

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def load_data(rootfile, treefile, embeddingfile):
    trees_file = rootfile + treefile

    with open(trees_file, "rb") as fh:
        trees = pickle.load(fh)

    # id = list(set(id))
    embedding_file = rootfile + embeddingfile
    with open(embedding_file, "rb") as fh:
        embeddings, lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    return trees, embeddings, lookup, num_feats


def similarity_class(model_dict, codestr1, codestr2, word2vec):
    """train data"""

    conv_feature = 600
    USE_CPU = 1

    # trees, embeddings, lookup, num_feats = load_data(args.path, '/C_trees_58.pkl','/C_embedding_58.pkl')
    embeddings = np.zeros(
        (word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32"
    )
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    num_feats = word2vec.vectors.shape[1]

    w_t = truncated_normal_(
        torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)
    ).cuda()
    w_l = truncated_normal_(
        torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)
    ).cuda()
    w_r = truncated_normal_(
        torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)
    ).cuda()
    init = truncated_normal_(
        torch.zeros(
            conv_feature,
        ),
        std=math.sqrt(2.0 / num_feats),
    ).cuda()
    b_conv = torch.tensor(init).cuda()
    ##!!!
    w_h = truncated_normal_(
        torch.zeros((conv_feature, 1)), std=1.0 / math.sqrt(conv_feature)
    ).cuda()
    b_h = truncated_normal_(
        torch.zeros(
            1,
        ),
        std=math.sqrt(2.0 / conv_feature),
    ).cuda()

    p1 = torch.tensor(num_feats).cuda()
    # p2 = torch.tensor(len(id)).cuda()
    p2 = torch.tensor(1).cuda()

    p3 = torch.tensor(conv_feature).cuda()

    model = TBCNNCLassifier(
        MAX_TOKENS,
        EMBEDDING_DIM,
        p1,
        p2,
        p3,
        w_t,
        w_l,
        w_r,
        b_conv,
        w_h,
        b_h,
        USE_CPU,
        embeddings,
    )
    model.load_state_dict(model_dict)

    if USE_CPU:
        model.cpu()

    node1, children1 = general_tbcnndata(codestr1, word2vec)
    node2, children2 = general_tbcnndata(codestr2, word2vec)
    output = model(node1, children1, node2, children2)

    return output
