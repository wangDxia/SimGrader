#!/usr/bin/python3
import sys
from math import log2


def get_HalsteadMetrics(path):
    operatorsFileName = "operators"
    programFileName = path

    operators = {}
    operands = {}

    with open(operatorsFileName) as f:
        for op in f:
            operators[op.replace("\n", "")] = 0

    isAllowed = True

    with open(programFileName) as f:
        for line in f:
            line = line.strip("\n").strip(" ")

            if line.startswith("/*"):
                isAllowed = False

            if (
                (not line.startswith("//"))
                and isAllowed == True
                and (not line.startswith("#"))
            ):
                for key in operators.keys():
                    operators[key] = operators[key] + line.count(key)
                    line = line.replace(key, " ")
                for key in line.split():
                    if key in operands:
                        operands[key] = operands[key] + 1
                    else:
                        operands[key] = 1

            if line.endswith("*/"):
                isAllowed = True

    n1, N1, n2, N2 = 0, 0, 0, 0

    for key in operators:
        if operators[key] > 0:
            if key not in ")}]":
                n1, N1 = n1 + 1, N1 + operators[key]

    for key in operands.keys():
        if operands[key] > 0:
            n2, N2 = n2 + 1, N2 + operands[key]

    val = {
        "N": N1 + N2,
        "n": n1 + n2,
        "V": (N1 + N2) * log2(n1 + n2),
        "D": n1 * N2 / 2 / n2,
    }
    val["E"] = val["D"] * val["V"]
    val["L"] = val["V"] / val["D"] / val["D"]
    val["I"] = val["V"] / val["D"]
    val["T"] = val["E"] / (18)
    val["N^"] = n1 * log2(n1) + n2 * log2(n2)
    val["L^"] = 2 * n2 / N2 / n1

    return [val["N"], val["n"], val["V"], val["D"], val["E"], val["T"]]


# get_HalsteadMetric('D:/code.c')
