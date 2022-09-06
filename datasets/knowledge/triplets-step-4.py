import pandas as pd
import os
import pickle
import numpy as np
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

CUI_STR_path = './umls_map/CUI_STR.csv'
CUI_STR = pd.read_csv(CUI_STR_path)
CUI2STR = CUI_STR.drop_duplicates(subset=['CUI']).loc[:, ['CUI', 'STR']].set_index('CUI').to_dict()['STR']

KG_PATH = f'./{args.task_name}-triplets'
os.makedirs(KG_PATH, exist_ok=True)

with open('./umls_map/fine_grained_map.pkl', mode='rb') as f:
    fine_grained_map = pickle.load(f)

CUI2idx = fine_grained_map['CUI2idx']
RELRL2idx = fine_grained_map['RELRL2idx']
# write entity
with open(f'{KG_PATH}/entity2id.txt', 'w') as f:
    f.write(f'{len(CUI2idx)}\n')
    for ent, idx in CUI2idx.items():
        f.write(f'{ent}\t{idx}\n')
# write relation
with open(f'{KG_PATH}/relation2id.txt', 'w') as f:
    f.write(f'{len(RELRL2idx) + 3}\n')
    # REL + RL + [token2CUI, CUI2STY ]
    for idx, rel in enumerate(list(RELRL2idx.keys()) + ['token2CUI', 'token2STY', 'CUI2STY']):  # 这就是为啥前面词向量加3的原因
        f.write(f'{rel}\t{idx}\n')

import json

temp = list(RELRL2idx.keys()) + ['token2CUI', 'token2STY', 'CUI2STY']
umls_rel2id = {key: value for value, key in enumerate(temp)}
with open('umls_rel2id_dict.json', encoding='utf-8', mode='w') as f:
    json.dump(umls_rel2id, f, indent=4, ensure_ascii=False)

    CUI_REL_CUI_RELA = pd.read_csv('./umls_map/CUI_REL_CUI_RELA.csv')
    STYRL1_RL_STYRL2_LS = pd.read_csv('./umls_map/STYRL_RL_STYRL.csv')

    CUI_df = CUI_REL_CUI_RELA.loc[:, ['CUI1', 'REL', 'CUI2']].rename(
        columns={'CUI1': 'subject', 'REL': 'relation', 'CUI2': 'object'})
    STY_df = STYRL1_RL_STYRL2_LS.rename(columns={'STY-RL1': 'subject', 'RL': 'relation', 'STY-RL2': 'object'}).loc[:,
             ['subject', 'relation', 'object']]
    STY_df = STY_df.loc[(STY_df.subject.isin(CUI2idx)) & (STY_df.object.isin(CUI2idx))]  # 我感觉这里也没有匹配上没啥用
    # concat
    CUInSTY_df = pd.concat([CUI_df, STY_df], axis=0)
    CUInSTY_df = CUInSTY_df.drop_duplicates()

    CUInSTY_df['subject'] = CUInSTY_df['subject'].apply(lambda x: CUI2idx[x])
    CUInSTY_df['object'] = CUInSTY_df['object'].apply(lambda x: CUI2idx[x])
    CUInSTY_df['relation'] = CUInSTY_df['relation'].apply(lambda x: RELRL2idx[x])  # 这里每一项已经都转化成索引了
    np.random.seed(42)
    indices = np.arange(len(CUInSTY_df))
    np.random.shuffle(indices)

    num_val = num_test = len(indices) // 10
    train_indices = indices[:-num_val - num_test]
    val_indices = indices[-num_val - num_test:-num_test]
    test_indices = indices[-num_test:]

    with open(f'{KG_PATH}/train2id.txt', 'w') as f:
        f.write(f'{len(train_indices)}\n')
        for row in CUInSTY_df.iloc[train_indices].itertuples():
            s = row.subject
            o = row.object
            r = row.relation
            f.write(f'{s}\t{o}\t{r}\n')

    with open(f'{KG_PATH}/valid2id.txt', 'w') as f:
        f.write(f'{len(val_indices)}\n')
        for row in CUInSTY_df.iloc[val_indices].itertuples():
            s = row.subject
            o = row.object
            r = row.relation
            f.write(f'{s}\t{o}\t{r}\n')

    with open(f'{KG_PATH}/test2id.txt', 'w') as f:
        f.write(f'{len(test_indices)}\n')
        for row in CUInSTY_df.iloc[test_indices].itertuples():
            s = row.subject
            o = row.object
            r = row.relation
            f.write(f'{s}\t{o}\t{r}\n')
