import os
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
np.random.seed(42)
random.seed(42)
root_path = '../cache/BCRE-data-org.json'

data = json.load(open(root_path, encoding='utf-8', mode='r'))

check = {}
non_check = {}
exception = {}
triggers = []
gold_count = 0
remove_count = 0
max_len = 0
max_non_len = 0
sss_dict = {}
e_type_set = set()
cache = []
for key, line in data.items():
    events = line['inter_events'] + line['intra_events']
    for event in events:
        if event[1].startswith('Causality'):
            item_E1 = event[2].split('-|-')[0]
            item_E2 = event[3].split('-|-')[0]
            if item_E1 in ['Cause', 'Effect'] and item_E2 in ['Cause', 'Effect']:
                triplet = None
                if item_E1 == 'Cause' and item_E2 == 'Effect':
                    triplet = f'E1 precedes E2'
                else:
                    triplet = f'E2 precedes E1'

                idx_1 = int(event[1].split('-|-')[-1])
                idx_2 = int(event[2].split('-|-')[-1])
                idx_3 = int(event[3].split('-|-')[-1])
                # max_len_ = max([abs(idx_1 - idx_2) + 1, abs(idx_1 - idx_3) + 1, abs(idx_2 - idx_3) + 1])
                max_len_ = abs(idx_2 - idx_3) + 1

                if max_len_ > 2:  # inter sentences more than 2 are removed
                    continue

                check[f'{key}&{event[0]}'] = event + [triplet]
                gold_count += 1
                if max_len_ > max_len:
                    max_len = max_len_
            else:
                idx_1 = int(event[1].split('-|-')[-1])
                idx_2 = int(event[2].split('-|-')[-1])
                idx_3 = int(event[3].split('-|-')[-1])
                # max_len_ = max([abs(idx_1 - idx_2) + 1, abs(idx_1 - idx_3) + 1, abs(idx_2 - idx_3) + 1])
                max_len_ = abs(idx_2 - idx_3) + 1
                if max_len_ > 2:
                    continue
                non_check[f'{key}&{event[0]}'] = event + ["None"]
                remove_count += 1
                if max_len_ > max_non_len:
                    max_non_len = max_len_
        else:
            if event[-1] == 0 and len(event) > 3 and not event[1].startswith("Process"):
                exception[f'{key}&{event[0]}'] = event + ["None", key]

print('the number of gold relations: ', gold_count)
print('the number of remove relations: ', remove_count)
print('max label span of sentence in a document: ', max_len)
print('max none span of sentence in a document: ', max_non_len)

event_pair = []

for head_info in exception.values():
    head_start = int(head_info[1].split("-|-")[-1])
    for tail_info in exception.values():
        tail_start = int(tail_info[1].split("-|-")[-1])
        if 0 <= tail_start - head_start <= 1 and head_info != tail_info and tail_info[-1] == head_info[-1]:
            event_pair.append([head_info, tail_info])

len_list = []

# 根据因果关系分
count = 1
pos_instances = []
exp_instances = []
pos_E1ToE2_sent_lens = []
pos_E2ToE1_sent_lens = []
check_list = []
for key, value in check.items():
    doc_id = key.split('&')[0]
    doc_sents = [item['words'].split() for item in data[doc_id]['base']]
    left_part = value[2].split('-|-')
    right_part = value[3].split('-|-')
    if [value[2], value[3]] in check_list:
        check_list.append([value[2], value[3]])
    left_sent_idx, right_sent_idx = int(left_part[-1]), int(right_part[-1])
    if left_sent_idx != right_sent_idx:
        if left_sent_idx < right_sent_idx:
            aux_len = len(doc_sents[left_sent_idx])
            e1_span = {'span': left_part[1].split(), 'start': int(left_part[2]), 'end': int(left_part[3]),
                       'label': left_part[0], 'idx': left_sent_idx}

            e2_span = {'span': right_part[1].split(), 'start': int(right_part[2]) + aux_len,
                       'end': int(right_part[3]) + aux_len, 'label': right_part[0], 'idx': right_sent_idx}
            tokens = doc_sents[left_sent_idx] + doc_sents[right_sent_idx]
        else:
            aux_len = len(doc_sents[right_sent_idx])
            e1_span = {'span': right_part[1].split(), 'start': int(right_part[2]),
                       'end': int(right_part[3]), 'label': right_part[0], 'idx': right_sent_idx}
            e2_span = {'span': left_part[1].split(), 'start': int(left_part[2]) + aux_len,
                       'end': int(left_part[3]) + aux_len, 'label': left_part[0], 'idx': left_sent_idx}
            tokens = doc_sents[right_sent_idx] + doc_sents[left_sent_idx]
    else:
        tokens = doc_sents[left_sent_idx]
        e1_span = {'span': left_part[1].split(), 'start': int(left_part[2]), 'end': int(left_part[3]),
                   'label': left_part[0], 'idx': left_sent_idx}

        e2_span = {'span': right_part[1].split(), 'start': int(right_part[2]),
                   'end': int(right_part[3]), 'label': right_part[0], 'idx': right_sent_idx}
    if left_sent_idx != right_sent_idx:
        cross_sentence = True
    else:
        cross_sentence = False

    temp = {"annotator-id": f'{doc_id}&p{count}',
            "cross-sentence": cross_sentence,
            "e1-sentence": doc_sents[e1_span['idx']],
            "e1-sentence-index": e1_span['idx'],
            "e1-label": e1_span['label'],
            "e1-start": e1_span['start'],
            "e1-end": e1_span['end'],
            "e1-trigger-start": e1_span['start'],
            "e1-trigger-end": e1_span['end'],
            "e1-tokens": tokens[e1_span['start']:e1_span['end']],
            "e1-trigger": " ".join(tokens[e1_span['start']:e1_span['end']]),
            "e2-sentence": doc_sents[e2_span['idx']],
            "e2-sentence-index": e2_span['idx'],
            "e2-label": e2_span['label'],
            "e2-start": e2_span['start'],
            "e2-end": e2_span['end'],
            "e2-trigger-start": e2_span['start'],
            "e2-trigger-end": e2_span['end'],
            "e2-tokens": tokens[e2_span['start']:e2_span['end']],
            "e2-trigger": " ".join(tokens[e2_span['start']:e2_span['end']]),
            "relation": value[-1],
            "text": ' '.join(tokens),
            "tokens": tokens,
            "org": "Cause-Effect"}
    len_list.append(len(e1_span['span']))
    len_list.append(len(e2_span['span']))
    count += 1
    if value[-1].startswith("E1"):
        pos_E1ToE2_sent_lens.append(len(tokens))
    else:
        pos_E2ToE1_sent_lens.append(len(tokens))
    pos_instances.append(temp)

neg_instances1 = []
for key, value in non_check.items():
    doc_id = key.split('&')[0]
    doc_sents = [item['words'].split() for item in data[doc_id]['base']]
    left_part = value[2].split('-|-')
    right_part = value[3].split('-|-')
    left_sent_idx, right_sent_idx = int(left_part[-1]), int(right_part[-1])
    if left_sent_idx != right_sent_idx:
        if left_sent_idx < right_sent_idx:
            aux_len = len(doc_sents[left_sent_idx])
            e1_span = {'span': left_part[1].split(), 'start': int(left_part[2]), 'end': int(left_part[3]),
                       'label': left_part[0], 'idx': left_sent_idx}

            e2_span = {'span': right_part[1].split(), 'start': int(right_part[2]) + aux_len,
                       'end': int(right_part[3]) + aux_len, 'label': right_part[0], 'idx': right_sent_idx}
            tokens = doc_sents[left_sent_idx] + doc_sents[right_sent_idx]
        else:
            aux_len = len(doc_sents[right_sent_idx])
            e1_span = {'span': right_part[1].split(), 'start': int(right_part[2]),
                       'end': int(right_part[3]), 'label': right_part[0], 'idx': right_sent_idx}
            e2_span = {'span': left_part[1].split(), 'start': int(left_part[2]) + aux_len,
                       'end': int(left_part[3]) + aux_len, 'label': left_part[0], 'idx': left_sent_idx}
            tokens = doc_sents[right_sent_idx] + doc_sents[left_sent_idx]
    else:
        tokens = doc_sents[left_sent_idx]
        e1_span = {'span': left_part[1].split(), 'start': int(left_part[2]), 'end': int(left_part[3]),
                   'label': left_part[0], 'idx': left_sent_idx}

        e2_span = {'span': right_part[1].split(), 'start': int(right_part[2]),
                   'end': int(right_part[3]), 'label': right_part[0], 'idx': right_sent_idx}

    if left_sent_idx != right_sent_idx:
        cross_sentence = True
    else:
        cross_sentence = False
    temp = {"annotator-id": f'{doc_id}&n{count}',
            "cross-sentence": cross_sentence,
            "e1-sentence": doc_sents[e1_span['idx']],
            "e1-sentence-index": e1_span['idx'],
            "e1-label": e1_span['label'],
            "e1-start": e1_span['start'],
            "e1-end": e1_span['end'],
            "e1-trigger-start": e1_span['start'],
            "e1-trigger-end": e1_span['end'],
            "e1-tokens": tokens[e1_span['start']:e1_span['end']],
            "e1-trigger": " ".join(tokens[e1_span['start']:e1_span['end']]),
            "e2-sentence": doc_sents[e2_span['idx']],
            "e2-sentence-index": e2_span['idx'],
            "e2-label": e2_span['label'],
            "e2-start": e2_span['start'],
            "e2-end": e2_span['end'],
            "e2-trigger-start": e2_span['start'],
            "e2-trigger-end": e2_span['end'],
            "e2-tokens": tokens[e2_span['start']:e2_span['end']],
            "e2-trigger": " ".join(tokens[e2_span['start']:e2_span['end']]),
            "relation": "None",
            "text": ' '.join(tokens),
            "tokens": tokens,
            "org": "Cause-Evidence"}
    assert temp['e1-trigger-start'] < temp['e1-trigger-end']
    assert temp['e2-trigger-start'] < temp['e2-trigger-end']
    assert temp['e1-start'] < temp['e1-end']
    assert temp['e2-start'] < temp['e2-end']

    count += 1
    if 3 <= len(temp["tokens"]) <= 85:
        neg_instances1.append(temp)


len_list = []
neg_instances2 = []
for e_head, e_end in event_pair:
    e_span = set()
    e1_info = {}
    doc_id = None
    for i, item in enumerate(e_head[1:-3]):
        type_, start_, end_, s_id_ = item.split("-|-")[1:]
        start, end, s_id = int(start_), int(end_), int(s_id_)
        e_span.add(int(start))
        e_span.add(int(end))
        if i == 0:
            e1_info["trigger_start"] = start
            e1_info["trigger_end"] = end
            e1_info["sent-idx"] = s_id
            e1_info["type"] = type_
            doc_id = e_head[-1]
    e1_info["start"] = min(e_span)
    e1_info["end"] = max(e_span)

    e_span = set()
    e2_info = {}
    for i, item in enumerate(e_end[1:-3]):
        type_, start_, end_, s_id_ = item.split("-|-")[1:]
        start, end, s_id = int(start_), int(end_), int(s_id_)
        e_span.add(int(start))
        e_span.add(int(end))
        if i == 0:
            e2_info["trigger_start"] = start
            e2_info["trigger_end"] = end
            e2_info["sent-idx"] = s_id
            e2_info["type"] = type_
    e2_info["start"] = min(e_span)
    e2_info["end"] = max(e_span)

    doc_sents = [item['words'].split() for item in data[doc_id]['base']]

    if e1_info["sent-idx"] == e2_info["sent-idx"]:
        tokens = doc_sents[e1_info["sent-idx"]]
    else:
        if e1_info["sent-idx"] < e2_info["sent-idx"]:
            tokens = doc_sents[e1_info["sent-idx"]] + doc_sents[e2_info["sent-idx"]]
            aux_len = len(doc_sents[e1_info["sent-idx"]])
            e2_info["trigger_start"] += aux_len
            e2_info["trigger_end"] += aux_len
            e2_info["start"] += aux_len
            e2_info["end"] += aux_len
        else:
            tokens = doc_sents[e2_info["sent-idx"]] + doc_sents[e1_info["sent-idx"]]
            aux_len = len(doc_sents[e2_info["sent-idx"]])
            e1_info["trigger_start"] += aux_len
            e1_info["trigger_end"] += aux_len
            e1_info["start"] += aux_len
            e1_info["end"] += aux_len
            temp = e1_info
            e2_info = e1_info
            e1_info = temp

    if e1_info["sent-idx"] != e2_info["sent-idx"]:
        cross_sentence = True
    else:
        cross_sentence = False
    temp = {"annotator-id": f'{doc_id}&n{count}',
            "cross-sentence": cross_sentence,
            "e1-sentence": doc_sents[e1_info['sent-idx']],
            "e1-sentence-index": e1_info['sent-idx'],
            "e1-label": e1_info['type'],
            "e1-start": e1_info['start'],
            "e1-end": e1_info['end'],
            "e1-trigger-start": e1_info['trigger_start'],
            "e1-trigger-end": e1_info['trigger_end'],
            "e1-tokens": tokens[e1_info['start']:e1_info['end']],
            "e1-trigger": " ".join(tokens[e1_info['trigger_start']:e1_info['trigger_end']]),
            "e2-sentence": doc_sents[e2_info['sent-idx']],
            "e2-sentence-index": e2_info['sent-idx'],
            "e2-label": e2_info['type'],
            "e2-start": e2_info['start'],
            "e2-end": e2_info['end'],
            "e2-trigger-start": e2_info['trigger_start'],
            "e2-trigger-end": e2_info['trigger_end'],
            "e2-tokens": tokens[e2_info['start']:e2_info['end']],
            "e2-trigger": " ".join(tokens[e2_info['trigger_start']:e2_info['trigger_end']]),
            "relation": "None",
            "text": ' '.join(tokens),
            "tokens": tokens,
            "org": "Event-Event"}
    count += 1
    assert e1_info['trigger_start'] < e1_info['trigger_end']
    assert e1_info['start'] < e1_info['end']
    assert e2_info['trigger_start'] < e2_info['trigger_end']
    assert e2_info['start'] < e2_info['end']

    if 3 <= len(temp["tokens"]) <= 85:
        neg_instances2.append(temp)

max_neg_num = 3140

sample_num = max_neg_num - len(neg_instances1)

skf = StratifiedKFold(n_splits=10)
t = [1 for _ in range(len(neg_instances2))]
target_list = []
for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(t)), t)):
   sample_num_ = sample_num // 10 + 1
   target_index = random.sample(list(test_index), sample_num_)
   target_list.extend(target_index)

neg_instances = neg_instances1 + [neg_instances2[i] for i in target_list[:sample_num]]

max_token_lens = []
for item in neg_instances:
    max_token_lens.append(len(item["tokens"]))

print("正例数: ", len(pos_E1ToE2_sent_lens) + len(pos_E2ToE1_sent_lens))
print("正例(E1-Precedes-E2)数: ", len(pos_E1ToE2_sent_lens))
print("正例(E1-Precedes-E2)最大句子长度: ", max(pos_E1ToE2_sent_lens))
print("正例(E1-Precedes-E2)最小句子长度: ", min(pos_E1ToE2_sent_lens))
print("正例(E1-Precedes-E2)平均句子长度: ", round(sum(pos_E1ToE2_sent_lens) / len(pos_E1ToE2_sent_lens), 1))
print("正例(E2-Precedes-E1)数: ", len(pos_E2ToE1_sent_lens))
print("正例(E2-Precedes-E1)最大句子长度: ", max(pos_E2ToE1_sent_lens))
print("正例(E2-Precedes-E1)最小句子长度: ", min(pos_E2ToE1_sent_lens))
print("正例(E2-Precedes-E1)平均句子长度: ", round(sum(pos_E2ToE1_sent_lens) / len(pos_E1ToE2_sent_lens)))
print("负例(None)数: ", len(max_token_lens))
print("负例(None)最大句子长度: ", max(max_token_lens))
print("负例(None)最小句子长度: ", min(max_token_lens))
print("负例(None)平均句子长度: ", round(sum(max_token_lens) / len(max_token_lens), 1))
# instances = pos_instances + neg_instances

fold_nums = 10

pos_num, neg_num = len(pos_instances) // (fold_nums - 1), len(neg_instances) // (fold_nums - 1)

field_dict = {i: [] for i in range(fold_nums)}

for i in range(fold_nums):
    start = i * pos_num
    end = min((i + 1) * pos_num, len(pos_instances))
    field_dict[i].extend(pos_instances[start:end])

for i in range(fold_nums):
    start = i * neg_num
    end = min((i + 1) * neg_num, len(neg_instances))
    field_dict[i].extend(neg_instances[start:end])

instances = []
for i in range(fold_nums):
    instances.extend(field_dict[i])

# instances = pos_instances + neg_instances
np.random.shuffle(instances)
with open("../std_data/annotations.json", encoding="utf-8", mode="w") as f:
    json.dump(instances, f, indent=4, ensure_ascii=False)
