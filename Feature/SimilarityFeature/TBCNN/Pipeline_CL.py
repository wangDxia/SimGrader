import pandas as pd
import os

# from prepare_data import get_blocks as func
from gensim.models.word2vec import Word2Vec
from pycparser import c_parser
import newtrees as Trees


class Pipeline:
    def __init__(self, root):

        self.root = root
        self.sources = None
        self.train_file_path = "data/train/train_.pkl"
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, path, out, option):
        train_path = self.root + out

        if os.path.exists(train_path) and option is "existing":
            source = pd.read_pickle(train_path)
        else:

            parser = c_parser.CParser()
            source = pd.read_pickle(self.root + path)
            source = source.drop(["index1", "index2", "TED", "t1s", "t2s"], axis=1)

            source["code1"] = source["code1"].apply(parser.parse)
            source["code2"] = source["code2"].apply(parser.parse)
            source.to_pickle(train_path)
        return source

    # generate block sequences with index representations
    def generate_block_seqs(self, trees, part):

        word2vec = Word2Vec.load("/root/process_data/newnode_w2v_" + "128").wv
        vocab = list(word2vec.key_to_index.keys())
        max_token = len(word2vec.key_to_index)

        ast1 = []
        ast2 = []
        # print(trees['code1'][0])
        trees["code1"] = trees["code1"].apply(Trees.parse)
        trees["code2"] = trees["code2"].apply(Trees.parse)
        for i in range(len(trees)):
            ast1.append(
                Trees.gen_onesamples(trees["code1"][i], vocab, max_token, None, None, 1)
            )
            ast2.append(
                Trees.gen_onesamples(trees["code2"][i], vocab, max_token, None, None, 1)
            )

        trees["AST1"] = ast1
        trees["AST2"] = ast2
        trees.to_pickle(self.root + "/" + part + "blocksnew.pkl")

    # run for processing data to train
    def run(self):
        print("parse source code...")
        train = pd.read_pickle("data/split/train_.pkl")
        val = pd.read_pickle("data/split/val_.pkl")
        test = pd.read_pickle("data/split/test_.pkl")

        print("generate block sequences...")
        self.generate_block_seqs(train, "train")
        self.generate_block_seqs(val, "dev")
        self.generate_block_seqs(test, "test")


ppl = Pipeline("data/split/")
ppl.run()
