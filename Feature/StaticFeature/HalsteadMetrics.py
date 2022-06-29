import pycparser
import sys
from itertools import accumulate
import math


def merge(dict1, dict2):
    if isinstance(dict2, dict):
        for key in dict2.keys():
            dict1[key] = dict1.get(key, 0) + dict2[key]
    else:
        print("error")


def parse_halstead(node):

    if str(type(node)).split("'")[1] == "tuple":
        if len(node) == 0:
            return ({}, {})
        elif len(node) == 2 and str(type(node[0])).split("'")[1] == "str":
            return parse_halstead(node[1])
        else:
            # print("came here")
            operators = {}
            operands = {}
            node_list = node
            for child in node_list:
                ops, opr = parse_halstead(child)
                merge(operators, ops)
                merge(operands, opr)
            return (operators, operands)

    if node is None:
        return ({}, {})
    elif str(type(node)) == "<class 'pycparser.c_ast.FileAST'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            # print(type(child))
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.FuncDef'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.FuncDecl'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.ArrayDecl'>":
        operators = {"[]": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.PtrDecl'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.ParamList'>":
        operators = {"{}": 1, "()": 1}  # , ', ;' : len(node.params) - 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.ExprList'>":
        operators = {"()": 1, ", ;": len(node.exprs) - 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.EnumeratorList'>":
        operators = {"{}": 1}  # , ', ;' : len(node.enumerators) - 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.DeclList'>":
        operators = {"{}": 1}  # , ', ;' : len(node.decls) - 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.InitList'>":
        operators = {"{}": 1}  # , ', ;' : len(node.exprs) - 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Cast'>":
        operators = {}
        operands = {"->": 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Compound'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.CompoundLiteral'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.ArrayRef'>":
        operators = {"[]": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.If'>":
        if node.iffalse is not None:
            operators = {"if": 1, "else": 1, "()": 1, "{}": 2}
            operands = {}
        else:
            operators = {"if": 1, "()": 1, "{}": 1}
            operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.For'>":
        operators = {"for": 1, "()": 1, "{}": 1, ", ;": 2}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.While'>":
        operators = {"while": 1, "()": 1, "{}": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.DoWhile'>":
        operators = {"dowhile": 1, "()": 1, "{}": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Switch'>":
        operators = {"switch": 1, "()": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Case'>":
        operators = {"case": 1, ":": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Continue'>":
        operators = {"continue": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Default'>":
        operators = {"default": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Return'>":
        operators = {"return": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Break'>":
        operators = {"break": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.EllipsisParam'>":
        operators = {"...": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.EmptyStatement'>":
        operators = {}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Decl'>":
        if (
            str(type(node.type)) == "<class 'pycparser.c_ast.Enum'>"
            and str(type(node.type.values))
            == "<class 'pycparser.c_ast.EnumeratorList'>"
        ):
            operators = {", ;": (len(node.type.values.enumerators) - 1)}
        elif str(type(node.init)) == "<class 'pycparser.c_ast.InitList'>":
            operators = {", ;": len(node.init.exprs) - 1}
        elif (
            str(type(node.type)) == "<class 'pycparser.c_ast.FuncDecl'>"
            and str(type(node.type.args)) == "<class 'pycparser.c_ast.ParamList'>"
        ):
            operators = {", ;": len(node.type.args.params) - 2}
        else:
            operators = {", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Enum'>":
        operators = {"enum": 2}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Enumerator'>":
        operators = {}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.FuncCall'>":
        operators = {", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Goto'>":
        operators = {"goto": 1, ", ;": 1}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.ID'>":
        operators = {}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.IdentifierType'>":
        operators = {str((node.names)[0]): 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Label'>":
        operators = {":", 1}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.NamedInitializer'>":
        operators = {str(node.name): 1}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Struct'>":
        operators = {"struct": 1, "{}": 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.StructRef'>":
        operators = {}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.TypeDef'>":
        operators = {str(node.name): 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.TypeDecl'>":
        if node.declname == "main":
            operators = {str(node.declname): 1}
            operands = {}
        else:
            operators = {}
            operands = {str(node.declname): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.TypeName'>":
        operators = {}
        operands = {str(node.declname): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Union'>":
        operators = {"union": 1, "{}": 1, ", ;": 1}
        operands = {str(node.name): 1}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.UnaryOp'>":
        operators = {str(node.op): 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.BinaryOp'>":
        operators = {str(node.op): 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.TernaryOp'>":
        if node.iffalse is not None:
            operators = {":": 1, "?": 1}
            operands = {}
        else:
            operators = {":": 1}
            operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Assignment'>":
        operators = {str(node.op): 1, ", ;": 1}
        operands = {}
        node_list = node.children()
        for child in node_list:
            oprt, oprn = parse_halstead(child)
            merge(operators, oprt)
            merge(operands, oprn)
        return (operators, operands)

    elif str(type(node)) == "<class 'pycparser.c_ast.Constant'>":
        return ({}, {str(node.value): 1})
    else:
        return ({}, {})


class HalsteadMetrics:
    def __init__(self, operators, operands):
        self.operators = operators
        self.operands = operands
        self.n1 = len(operators.keys())
        self.n2 = len(operands.keys())
        self.N1 = list(accumulate(list(operators.values())))[-1]
        self.N2 = list(accumulate(list(operands.values())))[-1]

    def programLength(self):
        N = self.N1 + self.N2
        N_ = math.log(self.n1, 2) + math.log(self.n2, 2)
        Nj = math.log(math.factorial(self.n1), 2) + math.log(math.factorial(self.n2), 2)
        Nb = self.n1 * (math.sqrt(self.n1)) + self.n2 * (math.sqrt(self.n2))
        Nc = self.n1 * (math.sqrt(self.n1)) + self.n2 * (math.sqrt(self.n2))
        Ns = ((self.n1 + self.n2) * math.log(self.n1 + self.n2, 2)) / 2
        N_return = N
        return N_return

    def programVocabulary(self):
        return self.n1 + self.n2

    def programVolume(self):
        return (self.programLength()) * (math.log(self.programVocabulary(), 2))

    def programDifficulty(self):
        return ((self.n1) / 2) * (self.N2) / (self.n2)

    def programLevel(self):
        return 1 / (self.programDifficulty())

    def programMinimumVolume(self):
        return self.programLevel() * self.programVolume()

    def programEffort(self):
        return self.programDifficulty() * self.programVolume()

    def languageLevel(self):
        return self.programVolume() * (1 / math.sqrt(self.programDifficulty()))

    def intelligenceContent(self):
        return (self.programVolume()) / (self.programDifficulty())

    def programmingTime(self):
        f = 60
        S = 18
        return self.programEffort() / (f * S)

    def showMetrics(self):

        return [
            self.programLength(),
            self.programVocabulary(),
            self.programVolume(),
            self.programDifficulty(),
            self.programEffort(),
            self.programmingTime(),
        ]


def get_HalsteadMetrics(path):

    ast = pycparser.parse_file(path, use_cpp=True)
    main = ast.children()[-1]

    listOfOperators = parse_halstead(main)[0]
    listOfOperands = parse_halstead(main)[1]
    hal_metric = HalsteadMetrics(listOfOperators, listOfOperands)
    hal_metric_data = hal_metric.showMetrics()
    return hal_metric_data


