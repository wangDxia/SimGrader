"""Parse trees from a data source."""
import pickle
import random
import sys
from collections import defaultdict
import pandas as pd
import os
from similarity_model.TBCNN.java_ast import JavaAST
from tqdm import tqdm
from javalang.ast import Node
from gensim.models.word2vec import Word2Vec
from pycparser import c_parser


def gen_onesamples(trees, vector_lookup, max_token, ind1, ind2, f):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    nodes = []
    children = []
    queue = [(trees, -1)]
    while queue:
        node, parent_ind = queue.pop(0)
        node_ind = len(nodes)
        queue.extend([(child, node_ind) for child in node["children"]])
        children.append([])
        if parent_ind > -1:
            children[parent_ind].append(node_ind)

        if node["node"] in vector_lookup:
            if vector_lookup.index(node["node"]) == ind1 and f == 0:
                nodes.append(ind2)
            else:
                nodes.append(vector_lookup.index(node["node"]))
        else:
            nodes.append(max_token)

    return nodes, children


def parse(codestr):

    """Parse trees with the given arguments."""
    # parser = c_parser.CParser()

    # ast_tree = parser.parse(codestr)
    sample, size = _traverse_tree(codestr)

    return sample


def split_data(pairspath, filepath, lang, ratio):
    data_path = filepath
    data = pd.read_pickle(pairspath)

    data_num = len(data)
    ratios = [int(r) for r in ratio.split(":")]
    train_split = int(ratios[0] / sum(ratios) * data_num)
    val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

    data = data.sample(frac=1, random_state=666)
    train = data.iloc[:train_split]
    dev = data.iloc[train_split:val_split]
    test = data.iloc[val_split:]

    def check_or_create(path):
        if not os.path.exists(path):
            os.mkdir(path)

    train_path = data_path + "train/"
    check_or_create(train_path)
    train_file_path = train_path + "train_.pkl"
    train.to_pickle(train_file_path)

    dev_path = data_path + "dev/"
    check_or_create(dev_path)
    dev_file_path = dev_path + "dev_.pkl"
    dev.to_pickle(dev_file_path)

    test_path = data_path + "test/"
    check_or_create(test_path)
    test_file_path = test_path + "test_.pkl"
    test.to_pickle(test_file_path)


def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {"node": _nodes_name(root), "children": []}
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)

        children = list(child for _, child in current_node.children())
        # children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        for child in children:
            child_json = {"node": _nodes_name(child), "children": []}

            current_node_json["children"].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def _nodes_name(node):
    """Get the name of a node."""
    if isinstance(node, JavaAST):
        if isinstance(node, tuple):
            return node[1].name
        else:
            # if node.children() == None:

            return node.name
    else:
        return type(node).__name__


def _name(node):

    if isinstance(node, JavaAST):
        return node.name
    else:
        return node.__class__.__name__


def get_token(node):
    token = ""
    if isinstance(node, str):
        # token = node
        token = "Token"
    elif isinstance(node, set):
        token = "Modifier"  # node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))
