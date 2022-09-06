import os
import json

os.makedirs('umls_map', exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

root_path = f'../{args.task_name}/cache/BCRE-umls-data-two.json'
map_path = './umls2smtype_dict.json'

json_data = json.load(open(root_path, encoding='utf-8', mode='r'))
umls2stype_dict = json.load(open(map_path, encoding='utf-8', mode='r'))
CUIs = set()
STY = set()

for line in json_data:
    umls_terms = line['umls']
    for item in umls_terms:
        CUIs.add(item['CUI'])
        for SemType in item['SemTypes']:
            STY.add(umls2stype_dict[SemType])


with open('umls_map/BCRE-umls-CUI.json', encoding='utf-8', mode='w') as f:
    json.dump(list(CUIs), f, indent=4, ensure_ascii=False)

with open('umls_map/BCRE-umls-STY.json', encoding='utf-8', mode='w') as f:
    json.dump(list(STY), f, indent=4, ensure_ascii=False)

import pandas as pd
import json
import pickle

path = r'F:\MetaMap\umls-2020AA-full\2020AA-full\UMLS\2020AA\NET\SRSTR'
lines = []
with open(path, encoding='utf-8', mode='r') as f:
    for line in f.readlines():
        lines.append(line.split('|')[:-1])
STYRL1_RL_STYRL2_LS = pd.DataFrame(lines, columns=['STY-RL1', 'RL', 'STY-RL2', 'LS'])
STYRL1_RL_STYRL2_LS.to_csv('umls_map/STYRL_RL_STYRL.csv', index=None)
print('finishing extracting STYRL_RL_STYRL.csv !')
# 读取MRSTY.RRF,也就是CUI和STY的转化映射文件
lines = []
path = r'F:\MetaMap\umls-2020AA-full\2020AA-full\UMLS\2020AA\META\MRSTY.RRF'
with open(path, encoding='utf-8', mode='r') as f:
    lines = [line.split('|')[:-1] for line in f.readlines()]
CUI_STY = pd.DataFrame(lines, columns=['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF'])
CUI_STY.to_csv('umls_map/CUI_STY.csv', index=None)
print('finishing extracting CUI_STY.csv !')
# 读取MRCONSO.RRF
import pandas as pd

path = r'F:\MetaMap\umls-2020AA-full\2020AA-full\UMLS\2020AA\META\MRCONSO.RRF'
lines = []
with open(path, encoding='utf-8', mode='r') as f:
    for line in f.readlines():
        lines.append([line.split('|')[0], line.split('|')[6], line.split('|')[14]])
CUI_STR = pd.DataFrame(lines, columns=['CUI', 'ISPREF', 'STR'])
CUI_STR.to_csv('umls_map/CUI_STR.csv', index=None)
print('finishing extracting CUI_STR.csv !')
# 读取最大的程序MRREL.RRF,也就是CUI1和CUI2之间的RL关系
# 1-step: 分割读取6G的原始数据

import os
import json

Bulk_num = 10000000
path = r'F:\MetaMap\umls-2020AA-full\2020AA-full\UMLS\2020AA\META\MRREL.RRF'
cache = 'umls_map/cache'
if not os.path.exists(cache):
    os.mkdir(cache)
lines = []
part_id = 0
cache_list = os.listdir(cache)
if len(cache_list) < 9:
    Total_num = 0
    with open(path, encoding='utf-8', mode='r') as f:
        for i, _ in enumerate(f.readlines(), 1):
            Total_num = i
    print('we find the total num of item in MRREL.RRF is: ', Total_num)
    with open(path, encoding='utf-8', mode='r') as f:
        for i, line in enumerate(f.readlines(), 1):
            lines.append([line.split('|')[0], line.split('|')[3], line.split('|')[4], line.split('|')[7]])
            if len(lines) == Bulk_num or i == Total_num:
                CUI1_REL_CUI2_RELA = pd.DataFrame(lines, columns=['CUI1', 'REL', 'CUI2', 'RELA'])
                CUI1_REL_CUI2_RELA.to_csv(os.path.join(cache, 'CUI1_REL_CUI2_RELA={}.csv'.format(part_id)), index=None)
                print('have saved the csv file: ', 'CUI1_REL_CUI2_RELA={}.csv'.format(part_id))
                part_id += 1
                lines = []

print('finishing split the CUI1_REL_CUI2_RELA ！')

from collections import Counter

print('数据集中从知识库映射的CUI(实体)数目: ', len(CUIs))

CUIs = set(CUIs)
UMLS_path = 'umls_map'
cache = os.path.join(UMLS_path, 'cache')
cache_list = os.listdir(cache)
csv_num = len(cache_list)
# 首先确定全局的cui_neighbor_counter的统计结果
all_CUIs = []
threshold = 35  # 原代码是35
for i, cache_name in enumerate(cache_list):
    cache_path = os.path.join(cache, cache_name)
    split_csv = pd.read_csv(cache_path)
    filter_split_csv = split_csv.loc[(split_csv.CUI1.isin(CUIs) | split_csv.CUI2.isin(CUIs))]
    all_CUIs.extend(filter_split_csv.CUI1.tolist() + filter_split_csv.CUI2.tolist())

cui_neighbor_counter = Counter(all_CUIs)
with open('./umls_map/base_cui_neighbor_counter.pkl', mode='wb') as f:
    pickle.dump(cui_neighbor_counter, f)
print('finishing extract base_cui_neighbor_counter.pkl !')

filtered_cuis = [cui for cui, count in cui_neighbor_counter.items() if count > threshold]  # 去除低频的cui
selected_cuis = CUIs.union(filtered_cuis)  # filter_split_csv 因为是或的关系,所以可能会引入别的cui

merge_csv = None
for i, cache_name in enumerate(cache_list):
    cache_path = os.path.join(cache, cache_name)
    split_csv = pd.read_csv(cache_path)
    filtered_split_csv = split_csv.loc[(split_csv.CUI1.isin(selected_cuis)) & (split_csv.CUI2.isin(selected_cuis))]
    if i == 0:
        merge_csv = filtered_split_csv
    else:
        merge_csv = pd.concat([merge_csv, filtered_split_csv], ignore_index=True)
merge_csv = merge_csv.drop_duplicates()
with open('umls_map/base_selected_CUI.pkl', mode='wb') as f:
    pickle.dump(selected_cuis, f)
print('finishing extract base_selected_CUI.pkl !')

print('知识库中与数据关联的CUI的数目: {}'.format(len(selected_cuis)))
print('融合后的CUI和CUI关系映射的CSV形状: {}'.format(merge_csv.shape))
final_MRREL_save = os.path.join(UMLS_path, 'CUI_REL_CUI_RELA.csv')
merge_csv.to_csv(final_MRREL_save, index=False)
print('finishing extract and merge the final CUI_REL_CUI_RELA.csv !')


from collections import defaultdict
CUI_REL_CUI_RELA = pd.read_csv(os.path.join('./umls_map/CUI_REL_CUI_RELA.csv'))
STYRL_RL_STYRL = pd.read_csv(os.path.join('./umls_map/STYRL_RL_STYRL.csv'))
CUI_STY = pd.read_csv(os.path.join('./umls_map/CUI_STY.csv'))
# construct REL to idx mapping
REL2idx = {REL: idx for idx, REL in enumerate(sorted(set(CUI_REL_CUI_RELA.REL)))}
idx2REL = {idx: REL for idx, REL in enumerate(sorted(set(CUI_REL_CUI_RELA.REL)))}
# construct RL to idx mapping
RL2idx = {RL: idx for idx, RL in enumerate(sorted(STYRL_RL_STYRL.RL.unique()))}  # unique 类似于字典
idx2RL = {idx: RL for idx, RL in enumerate(sorted(STYRL_RL_STYRL.RL.unique()))}
# construct CUI to idx mapping 这个地方的设定是非常奇怪的
CUI2idx = {cui: idx for idx, cui in enumerate(sorted(selected_cuis.union(set(CUI_STY.STY))))}
idx2CUI = {idx: cui for idx, cui in enumerate(sorted(selected_cuis.union(set(CUI_STY.STY))))}
# construct STY to idx mapping
STY2idx = {sty: idx for idx, sty in enumerate(sorted(set(CUI_STY.STY)))}
idx2STY = {idx: sty for idx, sty in enumerate(sorted(set(CUI_STY.STY)))}
# construct REL + RL to idx mapping
RELRL2idx = {relation: idx for idx, relation in enumerate(list(REL2idx.keys()) + list(RL2idx.keys()))}
idx2RELRL = {idx: relation for idx, relation in enumerate(list(REL2idx.keys()) + list(RL2idx.keys()))}

CUI_REL_CUI_map = CUI_REL_CUI_RELA.groupby(['CUI1', 'CUI2']).REL.apply(list).to_dict()
CUI_adj_list = defaultdict(set)
for CUI1, CUI2 in CUI_REL_CUI_map.keys():
    CUI_adj_list[CUI1].add(CUI2)
    CUI_adj_list[CUI2].add(CUI1)

all_STYs = sorted(set(CUI_STY.STY))
fine_grained_map = {'REL2idx': REL2idx, 'idx2REL': idx2REL, 'RL2idx': RL2idx, 'idx2RL': idx2RL,
                    'CUI2idx': CUI2idx, 'idx2CUI': idx2CUI, 'RELRL2idx': RELRL2idx, 'idx2RELRL': idx2RELRL,
                    'all_STYs': all_STYs, 'CUI_adj_list': CUI_adj_list}
with open(os.path.join('umls_map/fine_grained_map.pkl'), mode='wb') as f:
    pickle.dump(fine_grained_map, f)
print('保存细粒度映射字典...')
