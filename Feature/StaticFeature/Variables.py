import pycparser
import sys
from itertools import accumulate
import math
from gensim.models import Word2Vec
import enchant
import re


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
            operators = []
            operands = []
            node_list = node
            for child in node_list:
                ops, opr = parse_node(child)
                operators += ops
                operands += opr
            return operators, operands

    if node is None:
        return [], []

    elif str(type(node)) == "<class 'pycparser.c_ast.For'>":
        operators = []
        operands = []
        if node.init != None:
            if type(node.init.children()[0][1]) == pycparser.c_ast.ID:
                operands.append(node.init.children()[0][1].name)

            else:
                operands.append(node.init.children()[0][1].type.declname)
        else:
            operands.append(node.next.expr.name)
        node_list = node.children()
        for child in node_list:
            ops, opr = parse_node(child)
            operators += ops
            operands += opr
        return operators, operands

    elif str(type(node)) == "<class 'pycparser.c_ast.TypeDecl'>":
        if node.declname == "main":
            operators = []
            operands = []
        else:
            operators = []
            operators.append(str(node.declname))
            operands = []
        node_list = node.children()
        for child in node_list:
            ops, opr = parse_node(child)
            operators += ops
            operands += opr
        return operators, operands

    else:
        operators = []
        operands = []
        node_list = node.children()
        for child in node_list:
            ops, opr = parse_node(child)
            operators += ops
            operands += opr
        return operators, operands


def delete_var(var_coms, var_strus):
    if var_strus == []:
        return var_coms
    else:
        v = []
        for j in var_coms:
            if j not in list(set(var_strus)):

                v.append(j)
        return v


def get_Variables(path):

    ast = pycparser.parse_file(path)
    var = []
    for i in range(len(ast.children())):
        if str(type(ast.children()[i][1])) == "<class 'pycparser.c_ast.FuncDef'>":
            var_com, var_stru = parse_node(ast.children()[i])
            var += delete_var(var_com, var_stru)

    return var


def split_var(var):
    var_l = []
    v1 = []
    for i in var:
        v1 += i.split("_")
    v2 = []
    for i1 in v1:
        v2 += i1.split("-")

    for i2 in v2:
        v3 = re.findall("[A-Z][^A-Z]*", i2)
        if v3 == []:
            var_l.append(i2)
        else:
            var_l += v3

    return var_l


def check_words(var_list):
    dict = enchant.Dict("en_US")
    if len(var_list) == 1:
        if len(var_list[0]) == 1:
            return False
        elif dict.check(var_list[0]):
            return True
        else:
            return False

    for i in var_list:
        if i != "":
            if len(i) != 1 and dict.check(i):
                return True
        else:
            return False
    return False


def cal_rate(path):
    var = get_Variables(path)
    count = 0
    for i in var:
        vl = split_var([i])
        if check_words(vl):
            count += 1

    return count / len(var)


