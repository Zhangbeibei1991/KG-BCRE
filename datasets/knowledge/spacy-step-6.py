import json
import pickle
import torch
import json

import spacy
from spacy.tokens import Doc

nlp = spacy.load('en_core_sci_md')


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

# 因为有停用词的缘故, 所有原来处理的依存有的是不对的,
# 这里就用scispacy重新处理了一下依存

class myTokenizer():
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = []
        split_words = text.split()
        # print(split_words)

        words.extend([w for w in split_words if w != ''])

        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp.tokenizer = myTokenizer(nlp.vocab)

with open(f'../{args.task_name}/cache/BCRE-kg-data-three.json', encoding='utf-8', mode='r') as f:
    data = json.load(f)

with open('umls_rel2id_dict.json', encoding='utf-8', mode='r') as f:
    umls_rel2id = json.load(f)

len_umls_rels = len(umls_rel2id)
graphs = []
edges = []
for line in data:
    tokens = ' '.join(line['tokens'])
    tokens_scispacy = nlp(tokens)
    dep_line = []
    src, dst = [], []
    for token in tokens_scispacy:
        for child in token.children:
            dep_line.append((token.i, child.i, token.dep_))
            src.append(token.i)
            dst.append(child.i)

    if len(dep_line) == 0:
        src.append(0)
        dst.append(0)

    kg_data = line['kg-data']
    kg_data['nodes'] = torch.LongTensor(kg_data['nodes']).cuda()
    edge_index = kg_data['edge_index']
    edge_attr = kg_data['edge_attr']

    edge_attr_onehot = []
    for attr in edge_attr:
        edge_onehot = [0] * len_umls_rels
        for item in attr:
            edge_onehot[umls_rel2id[item]] = 1
        edge_attr_onehot.append(edge_onehot)

    attr_tensor = torch.LongTensor(edge_attr_onehot).cuda()

    line['kg-data']['attr_th'] = attr_tensor

with open(f'../{args.task_name}/std_data/BCRE-KG-data.pkl', mode='wb') as f:
    pickle.dump(data, f)
