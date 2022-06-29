from deal_qustion_data import *
import numpy as np
from IdentifierCount import get_IdentifierCount
import pandas as pd
from zss import Node, simple_distance
from pycparser import c_parser, c_ast
import json
import C_parser
import random
import re
import time
import Java_parser
from tqdm import tqdm


def json_to_tree(toplevel):
    prog = Node("toplevel")

    def helper(obj):
        if isinstance(obj, list):
            node = Node(obj[0])
            for kid in obj[1:]:
                node.addkid(helper(kid))
            return node
        else:
            return Node(obj)

    for fun in toplevel:
        prog.addkid(helper(fun))

    return prog


def parser_C(codestr):
    cparser = c_parser.CParser()
    tree = cparser.parse(codestr)
    return tree


def getProblemCount(data):
    problem_id_dict = {}
    problem_id = list(data["problem_id"])
    lan = list(data["language"])
    for i in range(len(problem_id)):
        p = problem_id[i]
        l = lan[i]
        if (p not in problem_id_dict) and l == "C":
            problem_id_dict[p] = 1
        elif (p in problem_id_dict) and l == "C":
            problem_id_dict[p] += 1
    problem = []
    for i in problem_id_dict.keys():
        if problem_id_dict[i] >= 300:
            problem.append(i)

    return problem


def isExit(codestr, codelist):
    for i in codelist:
        if codestr == i:
            return False

    return True


def clean_C(codestr):

    c = codestr
    regex_1 = r"(//[^\n]+)|(/\*.+?\*/)|(//)"

    pattern1 = re.compile(regex_1, re.DOTALL)
    # res1 = re.findall(pattern1, c)
    code = re.sub(pattern1, "", c)

    strs = ""
    for i in code.split("\n"):
        if not i.strip().startswith("#"):
            strs = strs + i
            strs = strs + "\n"

    return strs


def delete_duplicate(data, problem):

    code = list(data["code"])
    lan = list(data["language"])
    all_correct = []
    all_error = []
    for j in problem:
        codelist = []
        index = []
        cocodelist = []
        coindex = []
        coast = []
        ast = []

        for i in range(len(list(data["problem_id"]))):

            if data["problem_id"][i] == j and lan[i] == "C":

                if data["result"][i] == 0:
                    if isExit(code[i], cocodelist):
                        c_str = data["codestr"][i]
                        cocodelist.append(c_str)
                        coast.append(data["ast"][i])
                        coindex.append(i)
                else:
                    if isExit(code[i], codelist):
                        c_str = data["codestr"][i]
                        codelist.append(c_str)
                        ast.append(data["ast"][i])
                        index.append(i)

        pro_data = {"code": codelist, "index": index, "ast": ast}
        copro_data = {"code": cocodelist, "index": coindex, "ast": coast}
        all_error.append(pro_data)
        all_correct.append(copro_data)

    data = {"error_data": all_error, "correct_data": all_correct}
    return data


def create_dataset(pro_data, rate, num):
    code1 = []
    index1 = []
    index2 = []
    code2 = []
    tree_ed = []
    ind1 = 0
    ind2 = 0
    all_index1 = []
    all_index2 = []
    tree_size1 = []
    tree_size2 = []

    for d in range(0, len(pro_data["error_data"])):
        err_index = random.sample(
            range(0, len(pro_data["error_data"][d]["ast"])),
            int(rate * len(pro_data["error_data"][d]["ast"])),
        )
        cor_index = random.sample(
            range(0, len(pro_data["correct_data"][d]["ast"])),
            int(rate * len(pro_data["correct_data"][d]["ast"])),
        )
        all_index1.append(cor_index)
        all_index2.append(err_index)

        list1 = [l1 for l1 in range(len(cor_index) - 1)]

        cn = 0
        start = time.time()
        for i in tqdm(range(len(list1))):

            list2 = random.sample(
                range(0, len(err_index)), min(int(num / len(cor_index)), len(err_index))
            )

            for k in range(len(list2)):
                # start = time.time()
                code1.append(pro_data["correct_data"][d]["code"][cor_index[list1[i]]])
                code2.append(pro_data["error_data"][d]["code"][err_index[list2[k]]])
                tree1 = pro_data["correct_data"][d]["ast"][cor_index[list1[i]]]
                tree2 = pro_data["error_data"][d]["ast"][err_index[list2[k]]]
                index1.append(list1[i] + ind1)
                index2.append(list2[k] + ind2)

                ast1, t_s1 = C_parser.parse(tree1)
                ast2, t_s2 = C_parser.parse(tree2)
                dis = simple_distance(ast1, ast2)
                tree_size1.append(t_s1)
                tree_size2.append(t_s2)
                tree_ed.append(dis)
                # end = time.time()
                # print(end - start)
                if cn == 10:
                    end = time.time()
                    print(end - start)
                    return 0
                cn += 1
        ind1 += len(cor_index)
        ind2 += len(err_index)

        data = {
            "code1": code1,
            "code2": code2,
            "index1": index1,
            "index2": index2,
            "TED": tree_ed,
            "t1s": tree_size1,
            "t2s": tree_size2,
        }
        allindex = {"corindex": all_index1, "errindex": all_index2}
        data = pd.DataFrame(data)
        allindex = pd.DataFrame(allindex)
        allindex.to_pickle("all_codeindex_new54.pkl")
        data.to_pickle("all_dataset_new54.pkl")
        print("num:" + str(d))

    return allindex, data.sample(frac=1).reset_index(drop=True)


def set_label(data):
    labellist = [25, 50, 100, 150]

    label = []
    for i in data["TED"]:
        flag = 0
        for j in range(len(labellist)):
            if i < labellist[j]:
                label.append(j)
                flag = 1
                break
        if flag == 0:
            label.append(len(labellist))
    data["label"] = label
    data.to_pickle("data/newdataset/all_data.pkl")
    return data


def set_ratelabel(data):
    labellist = [0.8]

    label = []
    for i in range(len(data["TED"])):
        flag = 0
        minnum = min([data["t1s"][i], data["t2s"][i]])
        r = data["TED"][i] / minnum * 1.0
        for j in range(len(labellist)):
            if r < labellist[j]:
                label.append(j)
                flag = 1
                break
        if flag == 0:
            label.append(len(labellist))
    data["label"] = label

    return data


def get_codestr(allindex, prodata):
    codestr = []
    for i in range(len(allindex["corindex"])):
        for j in allindex["corindex"][i]:
            codestr.append(prodata["correct_data"][i]["code"][j])
    for i in range(len(allindex["errindex"])):
        for j in allindex["errindex"][i]:
            codestr.append(prodata["error_data"][i]["code"][j])

    data = {"code": codestr}
    return pd.DataFrame(data)


def delete_index(data):
    strs = []
    for i in range(len(data["index1"])):

        s = str(data["index1"][i]) + ","
        s += str(data["index2"][i])
        if s not in strs:
            strs.append(s)
        else:
            data = data.drop([i])

    data = data.reset_index(drop=True)
    return data


if __name__ == "__main__":

    oj_data = pd.read_pickle("data/newdataset/oj_data.pkl")
    problem_id = getProblemCount(oj_data)
    pro_list = problem_id
    pro_data = delete_duplicate(oj_data, pro_list)
    allindex, data = create_dataset(pro_data, 1, 3000)
