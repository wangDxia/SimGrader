import pandas as pd
import os
import numpy as np
import clean_code
from tqdm import tqdm
from IdentifierCount import get_IdentifierCount
from Cppcheck import Cppcheck_text, get_infotype
from Cpplint_feature import cpplint_text, get_styleinfotype
from Cyclomatic_Com import get_complexity
from Halstead import get_HalsteadMetrics
from Variables import cal_rate
from Structure_count import get_Structurecount


def creat_file(codestr, rootpath, filepath):
    filename = os.path.join(rootpath, filepath)
    file = open(filename, "w")
    file.write(codestr)
    file.close()


datapath = "../data/newdataset/grade_data.pkl"


data = pd.read_pickle(datapath)

feature = {
    "code_str": [],
    "code_result": [],
    "IdentifierCount": [],
    "complexity": [],
    "unused_vars": [],
    "unused_lines": [],
    "style_count": [],
    "code_length": [],
    "code_vocabulary": [],
    "code_volume": [],
    "code_difficulty": [],
    "code_effort": [],
    "coding_time": [],
    "max_runtime": [],
    "min_runtime": [],
    "avg_runtime": [],
    "max_memory": [],
    "min_memory": [],
    "avg_memory": [],
    "test_cases": [],
}
identifier_count = []
Complexity = []
unused_vars = []
style_count = []
code_length = []
halstead = []
min_runtime = []
max_runtime = []
avg_runtime = []
min_memory = []
max_memory = []
avg_memory = []
code_result = []
code_str = []
testcase_c = []
blank_count = []
comment_count = []
var_r = []
for_count = []
while_count = []
if_count = []
error_list = []
for i in tqdm(range(0, len(data))):
    try:
        c_str, b_c, c_c = clean_code.clean_code(data["oricode"][i], "C")
        c_str, ic = get_IdentifierCount(c_str)
        creat_file(c_str, "D:/", "code.c")
        style_info = Cppcheck_text("D:/code.c")
        feature0 = get_infotype(style_info)
        lint_info = cpplint_text("D:/code.c")
        var_rate = cal_rate("D:/code.c")

        fs, ws, ifs = get_Structurecount("D:/code.c")
        feature1 = get_styleinfotype(lint_info)
        complexity = get_complexity("D:/code.c")
        h_metrics = get_HalsteadMetrics("D:/code.c")
        code_str.append(c_str)
        Complexity.append(complexity)
        identifier_count.append(ic)
        unused_vars.append(feature0)
        style_count.append(sum(feature1))
        halstead.append(h_metrics)
        min_runtime.append(min(data["real_time"][i]))
        max_runtime.append(max(data["real_time"][i]))
        avg_runtime.append(np.mean(data["real_time"][i]))
        min_memory.append(min(data["memory"][i]))
        max_memory.append(max(data["memory"][i]))
        avg_memory.append(np.mean(data["memory"][i]))
        code_result.append(data["result"][i])
        testcase_c.append(data["test_case_count"][i])
        blank_count.append(b_c)
        comment_count.append(c_c)
        var_r.append(var_rate)
        for_count.append(fs)
        while_count.append(ws)
        if_count.append(ifs)
    except:
        error_list.append(i)
        print(len(error_list))

feature["code_str"] = code_str
feature["code_result"] = code_result
feature["IdentifierCount"] = identifier_count
feature["complexity"] = Complexity
feature["style_count"] = style_count

unused_vars = list(map(list, zip(*unused_vars)))
feature["unused_vars"] = unused_vars[0][:]
feature["unused_lines"] = unused_vars[1][:]


halstead = list(map(list, zip(*halstead)))
feature["code_length"] = halstead[0][:]
feature["code_vocabulary"] = halstead[1][:]
feature["code_volume"] = halstead[2][:]
feature["code_difficulty"] = halstead[3][:]
feature["code_effort"] = halstead[4][:]
feature["coding_time"] = halstead[5][:]

feature["min_runtime"] = min_runtime
feature["max_runtime"] = max_runtime
feature["avg_runtime"] = avg_runtime
feature["min_memory"] = min_memory
feature["max_memory"] = max_memory
feature["avg_memory"] = avg_memory
feature["code_result"] = code_result
feature["test_cases"] = testcase_c
feature["var_rate"] = var_r
feature["comment_count"] = comment_count
feature["blank_count"] = blank_count
feature["for_count"] = for_count
feature["if_count"] = if_count
feature["while_count"] = while_count
Error_list = {"error": []}
Error_list["error"] = error_list
er = pd.DataFrame(Error_list)
er.to_pickle("../data/newdataset/er-13.pkl")
feature = pd.DataFrame(feature)
feature.to_pickle("../data/newdataset/feature_grade-13.pkl")
print("ok")
