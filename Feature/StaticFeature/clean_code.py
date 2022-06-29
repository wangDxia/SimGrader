import re
import sys
import os


def clean_code(codestr, language):
    if language == "C++" or language == "C":
        return clean_C(codestr)
    if language == "Python":
        return clean_Python(codestr)
    if language == "Java":
        return clean_Java(codestr)


def clean_C(codestr):
    regex_1 = r"(//[^\n]+)|(/\*.+?\*/)|(//)"

    pattern1 = re.compile(regex_1, re.DOTALL)
    res1 = re.findall(pattern1, codestr)
    code = re.sub(pattern1, "", codestr)
    str = ""
    for i in code.split("\n"):
        if not i.strip().startswith("#"):
            str = str + i
            str = str + "\n"
    c_c = getCommentCount(res1)
    b_c = getBlankLineCount(codestr)
    return str, b_c, c_c


def getCommentCount(res):
    c = 0
    for i in res:
        if i != "":
            c += 1

    return c


def getBlankLineCount(str):
    count = 0
    codestr = str.replace("\t", "")
    codelist = codestr.split("\n")
    for i in codelist:
        if i.strip() == "":
            count += 1

    return count

def clean_Python(codestr):
    code = ""
    return code

def clean_Java(codestr):
    code = ""
    return code

