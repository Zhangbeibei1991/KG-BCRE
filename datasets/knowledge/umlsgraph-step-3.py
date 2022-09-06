import pandas as pd
import json
import os
import pickle
from itertools import product
from tqdm import tqdm
from collections import deque, defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

with open(os.path.join('./umls_map/fine_grained_map.pkl'), mode='rb') as f:
    fine_grained_map = pickle.load(f)

'''
fine_grained_map = {'REL2idx':REL2idx,'idx2REL':idx2REL,'RL2idx':RL2idx,'idx2RL':idx2RL,
                    'CUI2idx':CUI2idx,'idx2CUI':idx2CUI,'RELRL2idx':RELRL2idx,'idx2RELRL':idx2RELRL,
                    'all_STYs':all_STYs,'CUI_adj_list':CUI_adj_list}
'''
all_CUIs = fine_grained_map['all_STYs']
CUI_adjacency_list = fine_grained_map['CUI_adj_list']
RELRL2idx = fine_grained_map['RELRL2idx']
CUI2idx = fine_grained_map['CUI2idx']
# 在所有节点中找到一个最小生成树
with open('./umls_map/base_cui_neighbor_counter.pkl', mode='rb') as f:
    CUI_neighbor_counter = pickle.load(f)
MST_adjacency_list = defaultdict(set)
all_CUIs = list(CUI_adjacency_list.keys())
visiting = set()
visited = set()

# iterate each i because some node may be disconnected in the KG
# 再建立生成树的过程,首先要保证所有的边都是只走一遍,参考哥尼斯堡问题
for i in tqdm(range(len(all_CUIs))):
    current_node = all_CUIs[i]
    if current_node not in visited:
        q = deque([current_node])

        while q:
            current_node = q.popleft()
            # haven't visited the current node before
            if current_node not in visited:
                visited.add(current_node)
                for neighbor_node in CUI_adjacency_list[current_node]:
                    if neighbor_node not in visiting:
                        visiting.add(neighbor_node)
                        MST_adjacency_list[current_node].add(neighbor_node)
                        MST_adjacency_list[neighbor_node].add(current_node)
                        q.append(neighbor_node)
assert sum([len(edge) for edge in MST_adjacency_list.values()]) < len(MST_adjacency_list) * 2, \
    (sum([len(edge) for edge in MST_adjacency_list.values()]), len(MST_adjacency_list) * 2)


class MSTFinder():
    def __init__(self):
        self.visited = set()
        # store all the CUIs in the found graph
        self.CUI2CUI_list = []
        self.visiting = set()

    def augment(self, CUI, n):
        '''
        Find the n neighbors with the maximum connection with CUI
        '''
        return sorted(CUI_adjacency_list[CUI], key=lambda x: CUI_neighbor_counter[x], reverse=True)[:n]

    def findMST(self, CUIs, MST_adjacency_list):

        # convert to set for faster lookup
        CUI_set = set(CUIs)
        for cur in CUIs:
            if cur not in self.visited and cur not in self.visiting:
                self.visiting.add(cur)
                for neighbor in MST_adjacency_list[cur]:
                    if neighbor not in self.visited and neighbor not in self.visiting:
                        self.visiting.add(neighbor)
                        # if can be reached, append current node and its neighbor to MST
                        if self.dfs(neighbor, CUI_set, MST_adjacency_list):
                            self.CUI2CUI_list.append((cur, neighbor))

        # convert MST nodes to edge index
        MST_CUI2idx = {CUI: idx for idx, CUI in enumerate(CUIs)}

        CUI_counter = len(MST_CUI2idx)
        edge_index = []
        for CUI1, CUI2 in self.CUI2CUI_list:
            if CUI1 not in CUIs:
                CUIs.append(CUI1)
                MST_CUI2idx[CUI1] = CUI_counter
                CUI_counter += 1
            if CUI2 not in CUIs:
                CUIs.append(CUI2)
                MST_CUI2idx[CUI2] = CUI_counter
                CUI_counter += 1

            edge_index.append((MST_CUI2idx[CUI1], MST_CUI2idx[CUI2]))

        return CUIs, edge_index

    def dfs(self, cur, CUI_set, MST_adjacency_list):
        '''
        args:
            cur: str, current CUI node.

        returns:
            True if found CUI in CUIS along the path
            False if no
        '''

        found = False
        self.visited.add(cur)

        for neighbor in MST_adjacency_list[cur]:

            if neighbor not in self.visited and neighbor not in self.visiting:
                self.visiting.add(neighbor)
                if self.dfs(neighbor, CUI_set, MST_adjacency_list):
                    found = True
                    self.CUI2CUI_list.append((cur, neighbor))

        return found or cur in CUI_set


class UMLSMap:
    def __init__(self):
        with open('./umls_map/BCRE-umls-CUI.json', encoding='utf-8', mode='r') as f:
            self.CUIs = json.load(f)
        with open('./umls_map/BCRE-umls-STY.json', encoding='utf-8', mode='r') as f:
            self.STYs = json.load(f)
        with open('./umls2smtype_dict.json', encoding='utf-8', mode='r') as f:
            self.umls2semtype_dict = json.load(f)
        self.save_map_dict()
        STYRL1_RL_STYRL2_LS = pd.read_csv('./umls_map/STYRL_RL_STYRL.csv')
        self.STYRL_RL_STYRL_map = STYRL1_RL_STYRL2_LS.groupby(['STY-RL1', 'STY-RL2']).RL.apply(list).to_dict()
        with open('./umls_map/map_dict.pkl', mode='rb') as f:
            self.map_dict, self.CUI_STY_map = pickle.load(f)

    def save_map_dict(self):
        file_names = os.listdir('./umls_map')
        if 'map_dict.pkl' not in file_names:
            CUI_REL_CUI_RELA = pd.read_csv('./umls_map/CUI_REL_CUI_RELA.csv')
            CUI_STY = pd.read_csv(os.path.join('./umls_map/CUI_STY.csv'))
            CUI_REL_CUI_map = CUI_REL_CUI_RELA.groupby(['CUI1', 'CUI2']).REL.apply(list).to_dict()
            CUI_STY_map = CUI_STY.groupby('CUI').STY.apply(list).to_dict()
            # CUI_REL_CUI_map = CUI_REL_CUI_RELA.groupby(['CUI1', 'CUI2']).RELA.apply( list).to_dict()  # 创建两个CUI之间的REL映射字典.
            with open('./umls_map/map_dict.pkl', mode='wb') as f:
                pickle.dump([CUI_REL_CUI_map, CUI_STY_map], f)

    def save_data(self):
        # 配对事件的关系
        root_path = f'../{args.task_name}/cache/BCRE-umls-data-two.json'
        # print(os.listdir("../"))
        data = json.load(open(root_path, encoding='utf-8', mode='r'))
        for s, line in tqdm(enumerate(data), desc='step-4'):
            kg_data = {}
            edge_attr = []
            edge_index = []
            kg_data['nodes'] = []

            tokens = line['tokens']
            umls = line['umls']
            CUIs = []
            STYs = {}
            token_positions = set()
            for item in umls:
                CUI = item['CUI']
                token_start = item['token_start']
                token_end = item['token_end']
                SemTypes = [self.umls2semtype_dict[abr_type] for abr_type in item['SemTypes']]
                span_list = list(range(token_start, token_end))
                for idx in span_list:
                    CUIs.append((idx, CUI))
                    token_positions.add(idx)
                for sty in SemTypes:
                    for idx in span_list:
                        if idx not in STYs:
                            STYs[idx] = []
                        STYs[idx].append(sty)
            token_positions = sorted(list(token_positions), key=lambda x: x, reverse=False)
            kg_data['nodes'] = token_positions
            kg_data['tokens'] = tokens
            kg_data['num_recognized_tokens'] = num_recognized_tokens = len(token_positions)

            # Step-1: 找到和token关联的CUI并建立关系
            CUI2CUI_idx = {}
            CUI_offset = num_recognized_tokens  # 算上token的偏移量再计算位置
            for token_idx, CUI in enumerate(CUIs):
                CUI = CUI[1]
                CUI_idx = token_idx + CUI_offset
                # if the current mapped CUI has not appear before in the current sentence
                if CUI not in CUI2CUI_idx:
                    CUI2CUI_idx[CUI] = CUI_idx
                else:
                    CUI_offset -= 1
                # token2CUI
                edge = ['token2CUI']

                # bi-directional
                edge_index.append([token_idx, CUI2CUI_idx[CUI]])
                edge_attr.append(edge)

                edge_index.append([CUI2CUI_idx[CUI], token_idx])
                edge_attr.append(edge)

            CUIs = list(CUI2CUI_idx.keys())
            # Step-2: 找到CUI和CUI之间的关系
            if len(CUIs) > 0:
                mst_finder = MSTFinder()
                CUIs, CUI_edge_index = mst_finder.findMST(CUIs, MST_adjacency_list)
                for CUIidx1, CUIidx2 in CUI_edge_index:
                    CUI1 = CUIs[CUIidx1]
                    CUI2 = CUIs[CUIidx2]
                    # edge = np.zeros(edge_attr_length)
                    edge = []
                    for REL in self.map_dict[(CUI1, CUI2)]:
                        if str(REL) != 'nan':
                            edge.append(REL)

                    edge_attr.append(edge)

                # convert CUI index to node index
                CUI_edge_index = [[idx1 + num_recognized_tokens, idx2 + num_recognized_tokens] for idx1, idx2 in
                                  CUI_edge_index]

                edge_index += CUI_edge_index

            # [tok, CUIs]
            kg_data['nodes'] += CUIs
            kg_data['CUI_length'] = len(CUIs)

            # Step-3: 找到CUI和STY之间的关系
            sentence_STY2idx = {}
            counter = len(kg_data['nodes'])  #
            for CUI_idx, CUI in enumerate(CUIs):
                # offset by number of recognized tokens
                CUI_idx += num_recognized_tokens
                if self.CUI_STY_map.get(CUI) != None:
                    # get all the mapped STYs
                    for STY in self.CUI_STY_map.get(CUI):
                        if STY not in sentence_STY2idx:
                            sentence_STY2idx[STY] = counter
                            counter += 1
                        # CUI2STY
                        edge = ['CUI2STY']
                        # bidirectional
                        edge_index.append([CUI_idx, sentence_STY2idx[STY]])
                        edge_attr.append(edge)
                        edge_index.append([sentence_STY2idx[STY], CUI_idx])
                        edge_attr.append(edge)

            # # Step-4: 找到和token关联的STY并建立关系
            # for token_idx, (idx, STY_list) in enumerate(STYs.items()):
            #     for sty in STY_list:
            #         if sty not in sentence_STY2idx:
            #             sentence_STY2idx[sty] = len(sentence_STY2idx)
            #         # token2CUI
            #         edge = ['token2STY']
            #
            #         # bi-directional
            #         edge_index.append([token_idx, sentence_STY2idx[sty]])
            #         edge_attr.append(edge)
            #
            #         edge_index.append([sentence_STY2idx[sty], token_idx])
            #         edge_attr.append(edge)

            # add STYs to the end
            temp = [STY for STY, idx in sorted(sentence_STY2idx.items(), key=lambda x: x[1])]
            kg_data['nodes'] += temp
            kg_data['STY_length'] = len(temp)

            # Step-4: 找到STY和STY对应的关系
            # STY 2 STY
            for STY1, STY2 in product(sentence_STY2idx.keys(), repeat=2):
                if (STY1, STY2) in self.STYRL_RL_STYRL_map:
                    edge_index.append([sentence_STY2idx[STY1], sentence_STY2idx[STY2]])
                    edge = []
                    # RL edge feature
                    for RL in self.STYRL_RL_STYRL_map[(STY1, STY2)]:
                        edge.append(RL)
                    edge_attr.append(edge)

            # map from CUI to idx
            for i in range(num_recognized_tokens, len(kg_data['nodes'])):
                kg_data['nodes'][i] = CUI2idx[kg_data['nodes'][i]]

            node_count = len(kg_data['nodes'])
            edge_node_count = len(kg_data['nodes'])
            try:
                assert node_count == edge_node_count
            except:
                print('Exception')
            kg_data['edge_index'] = edge_index
            kg_data['edge_attr'] = edge_attr
            line['kg-data'] = kg_data
            edge_index_set = set()
            for item in edge_index:
                edge_index_set.add(item[0])
                edge_index_set.add(item[1])
            try:
                assert node_count == len(edge_index_set)
            except:
                print('Exception')

        with open(f'../{args.task_name}/cache/BCRE-umls-data-three.json', encoding='utf-8', mode='w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    dd = UMLSMap()
    dd.save_data()
