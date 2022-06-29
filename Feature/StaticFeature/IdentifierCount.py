import pycparser
from pycparser import c_parser
import sys
from itertools import accumulate
import math


def merge(dict1, dict2):

    for key in dict2.keys():
        dict1[key] = dict1.get(key, 0) + dict2[key]


def parse_halstead(node):

    if str(type(node)).split("'")[1] == "tuple":
        if len(node) == 0:
            return ({}, {})
        elif len(node) == 2 and str(type(node[0])).split("'")[1] == "str":
            return parse_halstead(node[1])
        else:
            count = 0
            node_list = node
            for child in node_list:
                c = parse_halstead(child)
                count = c + count
                return count

    if node is None:
        return 0

    elif str(type(node)) == "<class 'pycparser.c_ast.IdentifierType'>":
        count = 1
        node_list = node.children()
        for child in node_list:
            c = parse_halstead(child)
            count += c
        return count
    else:
        count = 0
        node_list = node.children()
        for child in node_list:
            c = parse_halstead(child)
            count += c
        return count


def get_IdentifierCount(c_str):
    cparser = c_parser.CParser()
    try:
        ast = cparser.parse(c_str)
    except:
        index = c_str.find("int main")
        c_str = c_str[index:]
        ast = cparser.parse(c_str)
    main = ast.children()[-1]
    count = parse_halstead(ast) - len(ast.ext)
    return c_str, count


# print(get_IdentifierCount(c_str))
