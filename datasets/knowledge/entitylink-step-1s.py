import spacy
import scispacy
import argparse
import json
from tqdm import tqdm
from scispacy.linking import EntityLinker
from spacy.tokens import Doc

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

nlp = spacy.load("en_core_sci_md")
nlp.tokenizer = myTokenizer(nlp.vocab)
# This line takes a while, because we have to download ~1GB of data
# and load a large JSON file (the knowledge base). Be patient!
# Thankfully it should be faster after the first time you use it, because
# the downloads are cached.
# NOTE: The resolve_abbreviations parameter is optional, and requires that
# the AbbreviationDetector pipe has already been added to the pipeline. Adding
# the AbbreviationDetector pipe and setting resolve_abbreviations to True means
# that linking will only be performed on the long form of abbreviations.
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default="hpowell", help="task name")

args = parser.parse_args()

data_path = f"../{args.task_name}/cache/BCRE-std-data-one.json"

data = json.load(open(data_path, encoding="utf-8", mode="r"))


for line in tqdm(data,desc="entity-linker"):
    sentence = line["tokens"]

    doc = nlp(" ".join(sentence))

    umls = []

    for entity in doc.ents:
        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        linker = nlp.get_pipe("scispacy_linker")

        if len(entity._.kb_ents) > 0:
            bigger_weight = entity._.kb_ents[0]
            linker_result = linker.kb.cui_to_entity[bigger_weight[0]]
            CUI = linker_result.concept_id
            TUI = linker_result.types
            linker_dict = {"text": entity.text, "start": entity.start, "end": entity.end, "cui": CUI, "tui": TUI}

            if linker_dict not in umls:
                umls.append(linker_dict)
    assert len(umls) > 0
    line["umls"] = umls

with open(f'../{args.task_name}/cache/BCRE-umls-data-two.json', encoding='utf-8', mode='w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
