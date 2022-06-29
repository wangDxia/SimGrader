import pycparser
import sys
from itertools import accumulate
import math
from gensim.models import Word2Vec


def merge(dict1, dict2):
    if isinstance(dict2, dict):
        for key in dict2.keys():
            dict1[key] = dict1.get(key, 0) + dict2[key]
    else:
        print("error")


def parse_node(node):

    if str(type(node)).split("'")[1] == "tuple":
        if len(node) == 0:
            return [], []
        elif len(node) == 2 and str(type(node[0])).split("'")[1] == "str":
            return parse_node(node[1])
        else:
            # print("came here")
            fs = 0
            ws = 0
            i_fs = 0
            node_list = node
            for child in node_list:
                f, w, i_f = parse_node(child)
                fs += f
                ws += w
                i_fs += i_f

            return fs, ws, i_fs

    if node is None:
        return 0, 0, 0

    elif str(type(node)) == "<class 'pycparser.c_ast.For'>":
        fs = 1
        ws = 0
        i_fs = 0

        node_list = node.children()
        for child in node_list:
            f, w, i_f = parse_node(child)
            fs += f
            ws += w
            i_fs += i_f
        return fs, ws, i_fs

    elif str(type(node)) == "<class 'pycparser.c_ast.While'>":
        fs = 0
        ws = 1
        i_fs = 0

        node_list = node.children()
        for child in node_list:
            f, w, i_f = parse_node(child)
            fs += f
            ws += w
            i_fs += i_f
        return fs, ws, i_fs

    elif str(type(node)) == "<class 'pycparser.c_ast.If'>":
        fs = 0
        ws = 0
        i_fs = 1

        node_list = node.children()
        for child in node_list:
            f, w, i_f = parse_node(child)
            fs += f
            ws += w
            i_fs += i_f
        return fs, ws, i_fs

    else:
        fs = 0
        ws = 0
        i_fs = 0
        node_list = node.children()
        for child in node_list:

            f, w, i_f = parse_node(child)
            fs += f
            ws += w
            i_fs += i_f
        return fs, ws, i_fs


def delete_var(var_coms, var_strus):
    if var_strus == []:
        return var_coms
    else:
        v = []
        for i in list(set(var_strus)):
            for j in var_coms:
                if i != j:
                    v.append(j)
        return v


def get_Structurecount(path):

    ast = pycparser.parse_file(path)
    fs = 0
    ws = 0
    i_fs = 0
    for i in range(len(ast.children())):
        if str(type(ast.children()[i][1])) == "<class 'pycparser.c_ast.FuncDef'>":
            f, w, i_f = parse_node(ast.children()[i])
            fs += f
            ws += w
            i_fs += i_f

    return fs, ws, i_fs


# get_Structurecount('D:/code.c')
