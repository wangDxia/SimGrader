import pycparser
from pycparser import c_parser
import sys
import os

# sys.setrecursionlimit(100000)
dictionary = {}


def complexity_text(filepath):
    command = "lizard " + filepath + " 2>&1"
    r = os.popen(command)
    info = r.readlines()
    style_info = []
    for line in info:
        line = line.strip("\r\n")
        style_info.append(line)
    return style_info


def get_complexity(filepath):
    complexity_info = complexity_text(filepath)
    l = -len("@" + filepath)
    c = []
    for i in range(len(complexity_info)):
        if complexity_info[i][l:] == ("@" + filepath):
            num = complexity_info[i].split(" ")
            num = [n for n in num if n != ""]
            c.append(int(num[1]))

    return sum(c)


print(get_complexity("D:/new.c"))
