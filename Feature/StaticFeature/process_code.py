import clean_code
import pandas as pd
from tqdm import tqdm

datapath = "../data/newdataset/oj_data.pkl"

data = pd.read_pickle(datapath)
codestr = []

for i in tqdm(range(len(data))):
    c_str, b_c, c_c = clean_code.clean_code(data["code"][i], "C")
    codestr.append(c_str)

code = {"code": codestr}
code = pd.DataFrame(code)
code.to_pickle("../data/newdataset/codestr.pkl")
print("ok")
