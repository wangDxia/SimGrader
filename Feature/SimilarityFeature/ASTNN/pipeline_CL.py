import pandas as pd
import os


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
            from pycparser import c_parser

            parser = c_parser.CParser()
            source = pd.read_pickle(self.root + path)
            source = source.drop(["index1", "index2", "TED", "t1s", "t2s"], axis=1)

            source["code1"] = source["code1"].apply(parser.parse)
            source["code2"] = source["code2"].apply(parser.parse)
            source.to_pickle(train_path)
        return source

    # generate block sequences with index representations
    def generate_block_seqs(self, trees, part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root + "/node_w2v_" + "128").wv
        vocab = list(word2vec.key_to_index.keys())
        max_token = len(word2vec.key_to_index)

        def tree_to_index(node):
            token = node.token
            result = [vocab.index(token) if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees["code1"] = trees["code1"].apply(trans2seq)
        trees["code2"] = trees["code2"].apply(trans2seq)
        trees.to_pickle(self.root + "/" + part + "blocks_4.pkl")

    # run for processing data to train
    def run(self):
        print("parse source code...")
        # train = self.parse_source('train_set.pkl', 'train_.pkl', option='existing')
        # val = self.parse_source('val_set.pkl', 'val_.pkl', option='existing')
        test = self.parse_source("test_set.pkl", "test_.pkl", option="existing")

        print("generate block sequences...")
        # self.generate_block_seqs(train, 'train')
        # self.generate_block_seqs(val, 'dev')
        self.generate_block_seqs(test, "test")


ppl = Pipeline("data/split/")
ppl.run()
