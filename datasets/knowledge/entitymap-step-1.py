import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="biocause", help="task name")

args = parser.parse_args()

data_path = f"../{args.task_name}/cache/BCRE-std-data-one.json"

def process_output(output, split_start):
    phrases = output['AllDocuments'][0]['Document']['Utterances'][0]['Phrases']

    # stores all the mapped entities
    candidates = []
    for phrase in phrases:

        # starting position for this phrase
        phrase_start_pos = int(phrase['PhraseStartPos'])
        phrase_end_length = int(phrase['PhraseLength'])
        if len(phrase['Mappings']) == 0:
            continue
        # get the first mapping
        mapping = phrase['Mappings'][0]

        for candidate in mapping['MappingCandidates']:
            score = -int(candidate['CandidateScore'])
            match_text = ' '.join(candidate['MatchedWords'])
            match_len = int(candidate['ConceptPIs'][0]['Length'])
            # only append
            candidates.append(
                {
                    'CUI': candidate['CandidateCUI'],
                    'StartPos': int(candidate['ConceptPIs'][0]['StartPos']) + split_start,
                    'EndPos': int(
                        candidate['ConceptPIs'][0]['StartPos']) + split_start + match_len,
                    "MatchText": match_text,
                    'SemTypes': candidate['SemTypes'],
                    'CandidatePreferred': candidate['CandidatePreferred']
                }
            )

    return candidates


import subprocess
import json


def get_metamap_output(sentence, split_start):
    '''
    Given a sentence return the metamap best matching result (score, ID, term)
    sentence: str

    '''
    metamap_path = '/media/linus/000DE541000A7980/MetaMap/public_mm_linux_main_2020/public_mm/bin/metamap'
    p = subprocess.Popen(f'echo "{sentence}" | {metamap_path} --JSONn --I', stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    output = str(output, 'utf-8')

    output = output.split('\n')

    # no mapped entities
    if len(output) < 2:
        return None
    try:
        output = json.loads(output[1])
        output = process_output(output, split_start)
        return output
        # JSON Decoder
    except:
        return None

from joblib import Parallel, delayed
import time

def get_event_entities(tokens, left_idx, right_idx):
    token_str = ' '.join(tokens)
    part_sent_list = tokens[left_idx:right_idx]
    part_sent_str = ' '.join(part_sent_list)
    split_char = 0
    if left_idx == 0:
        split_char = 0
    else:
        split_char = len(' '.join(tokens[:left_idx])) + 1
    mapped_results = get_metamap_output(part_sent_str, split_char)
    mapped_seg_entities = []
    if mapped_results is not None:
        for t, item in enumerate(mapped_results):
            match_text = item['MatchText']
            if match_text in ['this, that, here, as, of, about, with, by, results, result']:
                continue
            if match_text.endswith('ly'):
                continue
            # 找tokenw位置
            char_start_pos = item['StartPos']
            char_end_pos = item['EndPos']
            # assert part_sent_str[char_start_pos:char_end_pos].lower() == match_text
            if char_end_pos < len(part_sent_str) and token_str[char_end_pos] != ' ':
                tag_flag = True
                while tag_flag and char_end_pos < len(part_sent_str):
                    if token_str[char_end_pos] != ' ':
                        char_end_pos += 1
                    else:
                        tag_flag = False
                # print(part_sent_str[char_start_pos:char_end_pos])
            if char_start_pos > 1 and token_str[char_start_pos - 1] != ' ':
                tag_flag = True
                # print(match_text)
                while tag_flag and char_start_pos < len(part_sent_str):
                    if token_str[char_start_pos - 1] != ' ':
                        char_start_pos -= 1
                    else:
                        tag_flag = False
                # print(part_sent_str[char_start_pos:char_end_pos])
            check_char_start, check_char_end = 0, 0
            token_start, token_end = 0, 0
            for s, token in enumerate(tokens):

                if char_start_pos == 0:
                    token_start = 0
                    break
                if check_char_start == char_start_pos:
                    token_start = s
                    break
                check_char_start += len(token)
                if s != len(tokens) - 1:
                    check_char_start += 1

            for s, token in enumerate(tokens):
                check_char_end += len(token)
                if s != len(tokens) - 1:
                    check_char_end += 1
                if check_char_end - 1 == char_end_pos:
                    token_end = s + 1
                    break
            if token_start == 0 and token_end != 0:
                token_start = token_end - 1
            elif token_start != 0 and token_end == 0:
                token_end = token_start + 1
            item['token'] = tokens[token_start:token_end]
            item['token_start'] = token_start
            item['token_end'] = token_end
            mapped_seg_entities.append(item)
    return mapped_seg_entities


def get_mapped_entities():
    data = json.load(open(data_path, encoding="utf-8", mode="r"))
    count = 0
    for i, line in tqdm(enumerate(data), desc='umls'):
        tokens = line['tokens']
        e1_span = line['e1-trigger-span']
        e2_span = line['e2-trigger-span']
        spans = sorted([e1_span[0], e1_span[1], e2_span[0], e2_span[1]], key=lambda x: x, reverse=False) + [len(tokens)]
        left_boundary = [0] + spans[:-1]
        right_boundary = spans
        sent_map_entities = []
        parallel = Parallel(7, backend="threading", verbose=0)
        temps = parallel(delayed(get_event_entities)(tokens, left_idx, right_idx) for (left_idx, right_idx) in zip(left_boundary, right_boundary))
        for temp in temps:
            sent_map_entities.extend(temp)
        count += len(sent_map_entities)
        line['umls'] = sent_map_entities
    print('抽取的UMLS实体数为: ', count)
    return data


data_ = get_mapped_entities()

with open(f'../{args.task_name}/cache/BCRE-umls-data-two.json', encoding='utf-8', mode='w') as f:
    json.dump(data_, f, indent=4, ensure_ascii=False)

# hpowell 抽取的UMLS实体数为:  14891
# biocause 抽取的UMLS实体数为:  51605