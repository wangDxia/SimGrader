import os
import re


class Cpplint_info:
    def __init__(self):
        self.use_info = [
            "whitespace/indent",
            "whitespace/comma",
            "whitespace/parens",
            "whitespace/braces",
            "whitespace/tab",
            "whitespace/operators",
            "whitespace/end_of_line",
            "whitespace/blank_line",
        ]
        self.num = 3


def creat_file(codestr, filepath, rootpath):

    filename = os.path.join(rootpath, filepath)
    with open(filename, "w") as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(codestr)


def cpplint_text(filepath):
    command = "cpplint " + filepath + " 2>&1"
    r = os.popen(command)
    info = r.readlines()
    style_info = []
    for line in info:
        line = line.strip("\r\n")
        style_info.append(line)
    return style_info


def get_styleinfotype(style_info):
    info = Cpplint_info()
    num_list = [0] * len(info.use_info)
    for s in style_info:
        index = re.findall(r"\[(.*?)\]", s)
        if len(index) == 2:
            for k in range(len(info.use_info)):
                if info.use_info[k] == index[0]:
                    if index[1].isdigit() and int(index[1]) >= info.num:
                        num_list[k] += 1
    return num_list

