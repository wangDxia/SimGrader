import os
import re


class Cppcheck_info:
    def __init__(self):
        self.use_info = ["uselessAssignmentArg", "unusedVariable"]


def Cppcheck_text(filepath):
    command = (
        'cppcheck --template={"{file},{line},{severity},{id},{message}"} --enable=all '
        + filepath
        + " 2>&1"
    )
    r = os.popen(command)
    info = r.readlines()
    style_info = []
    for line in info:
        line = line.strip("\r\n")
        style_info.append(line)
    return style_info


def get_infotype(style_info):
    info = Cppcheck_info()
    num_list = [0] * len(info.use_info)
    for s in style_info:
        index = s.split(",")
        if len(index) >= 4:
            for k in range(len(info.use_info)):
                if info.use_info[k] == index[3]:
                    num_list[k] += 1
    return num_list
