import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="hpowell", help="task name")

args = parser.parse_args()

root_path = f'../{args.task_name}/std_data/annotations.json'

root_data = json.load(open(root_path, encoding='utf-8', mode='r'))

root_statistics_dict = {}
for line in root_data:
    relation = line['relation']
    if relation not in root_statistics_dict:
        root_statistics_dict[relation] = 0
    root_statistics_dict[relation] += 1
print('ORGINAL instances'.center(30, '='))
for key, value in root_statistics_dict.items():
    print(key, value)
print('=' * 30)
print('Bug is abandoned ...')
statistics_dict = {}
BCRE_data_one = []
for line in root_data:
    relation = line['relation']
    if relation == 'Bug':  # 没考虑在内
        continue
    if relation not in ['E1 precedes E2', 'E2 precedes E1']:
        relation = 'None'
        line['relation'] = relation
    BCRE_data_one.append(line)
    if relation not in statistics_dict:
        statistics_dict[relation] = 0
    statistics_dict[relation] += 1
print('BCRE instances'.center(30, '='))
for key, value in statistics_dict.items():
    print(key, value)
print('=' * 30)
max_len = 0
std_data = []
for i, line in enumerate(BCRE_data_one):
    relation = line['relation']
    e1_start, e1_end = line['e1-start'], line['e1-end']
    e1_trigger_start, e1_trigger_end = line['e1-trigger-start'], line['e1-trigger-end']
    e1_trigger = line['e1-trigger']
    e2_start, e2_end = line['e2-trigger-start'], line['e2-trigger-end']
    e2_trigger_start, e2_trigger_end = line['e2-trigger-start'], line['e2-trigger-end']
    e2_trigger = line['e2-trigger']
    e1_label, e2_label = line['e1-label'], line['e2-label']
    e1_sentence_index = line['e1-sentence-index']
    e2_sentence_index = line['e2-sentence-index']
    # assert abs(e1_sentence_index - e2_sentence_index) <= 1
    sentence_list = None

    if args.task_name == "biocause":
        if e1_sentence_index == e2_sentence_index:
            sentence_list = line['tokens']
        else:
            sentence_list = line['tokens']
            # e1_txt = ' '.join(sentence_list[e1_trigger_start:e1_trigger_end])
            # e2_txt = ' '.join(sentence_list[e2_trigger_start:e2_trigger_end])
            assert ' '.join(sentence_list[e1_trigger_start:e1_trigger_end]) == e1_trigger
            assert ' '.join(sentence_list[e2_trigger_start:e2_trigger_end]) == e2_trigger
    elif args.task_name == "hpowell":
        if e1_sentence_index == e2_sentence_index:
            sentence_list = line['e1-tokens']
        else:
            len_sent_1 = len(line['e1-tokens'])
            e2_start, e2_end = e2_start + len_sent_1, e2_end + len_sent_1
            e2_trigger_start, e2_trigger_end = e2_trigger_start + len_sent_1, e2_trigger_end + len_sent_1
            sentence_list = line['e1-tokens'] + line['e2-tokens']
            assert ' '.join(sentence_list[e1_trigger_start:e1_trigger_end]) == e1_trigger
            assert ' '.join(sentence_list[e2_trigger_start:e2_trigger_end]) == e2_trigger

    # bert_token = ['[CLS]']
    # bert_span = []
    # offsets = []
    # bert_start, bert_end = 1, 0
    # for token in sentence_list:
    #     bert_tk = tokenizer.tokenize(token)
    #     bert_end = bert_start + len(bert_tk)
    #     bert_token.extend(bert_tk)
    #     bert_span.append((bert_start, bert_end))
    #     offsets.append(bert_start)
    #     bert_start = bert_end
    # bert_token.append('[SEP]')
    # bert_tk_id = tokenizer.encode(' '.join(sentence_list))
    # assert len(bert_token) == len(bert_tk_id)
    line_dict = {'tokens': sentence_list,
                 'e1-span': [e1_start, e1_end, e1_label],
                 'e2-span': [e2_start, e2_end, e2_label],
                 'e1-trigger-span': [e1_trigger_start, e1_trigger_end, e1_trigger],
                 'e2-trigger-span': [e2_trigger_start, e2_trigger_end, e2_trigger],
                 'relation': relation,
                 # 'bert-info': {'bert-tk-ids': bert_tk_id, 'bert-offsets': offsets,
                 #               'bert-spans': bert_span, 'bert-tokens': bert_token},
                 'idx': i}
    # max_len = max(max_len, len(bert_token))
    std_data.append(line_dict)
# print('max length: ', max_len)
os.makedirs(f'../{args.task_name}/cache')
with open(f'../{args.task_name}/cache/BCRE-std-data-one.json', mode='w', encoding='utf-8') as f:
    json.dump(std_data, f, indent=4, ensure_ascii=False)
