#  This is the cleaned version of `gen_camb_test.py` for sharing

import random, string
import json, jsonlines
import os, sys
import spacy
from tqdm import tqdm
from typing import Dict, List, Any
from .utils.sense_dicts import preprocess_definition

INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_PATH, "data")
# DATA_FOLDER = "/home/nlplab/kedy/NLP/AES/data"
FILE_CAM_DICT = "cambridge.word.888.json"
STOPWORDS_PATH = os.path.join(DATA_FOLDER, "stopwords.txt")
STOPWORDS = set([line.strip() for line in open(STOPWORDS_PATH)])

model_en = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat', 'custom'])


def format_prompt_input(word: str, sent: str, definitions: List) -> Dict:
    """Turns token and its dictionary definitions + token's context sentence
    into input prompt for WSD model.
    :param word: word to be disambiguated
    :param sent: context sentence of `word`
    :param definitions: List[Dict]. Output of `get_cambridge_dict_info()`
    :return: Dict[str]
    """
    def mark_target(sent: str, target: str) -> str:
        sent = model_en(sent)
        sent_info = {
                word.lemma_: {"idx": word.i, "text": word.text,}
                    for word in sent
                }
        try:
            lemma = target.lower()
            target_idx = sent_info[lemma]["idx"] # target.lower() should be the lemma
            target_orig_form = sent_info[lemma]['text']
            out_sent = [tok.text for tok in sent]
            out_sent[target_idx] = f'" {target_orig_form} "'
            out_sent = " ".join(out_sent)
            return out_sent
        except KeyError as ke:
            print(f"Key error with word {target}: {ke}")
            return None
        except IndexError as ie:
            print(f"Index error with word {target}: {ie}")
            return None

    for i, item in enumerate(definitions):
        # Probably because the model was fine-tuned using definition indices
        # that start with 1, we number the definitions starting with 1 also.
        item["numbered_def"] = f"({i + 1}) {item['data-en_def']}"

    context_sent = mark_target(sent, word)
    if context_sent is None:
        return None
    input_prompt = INPUT_TEMPLATE.format(
        word,
        ' " , " '.join([item["numbered_def"] for item in definitions][:-1]),
        [item["numbered_def"] for item in definitions][-1],  # "def1, ..., or def4"
        context_sent,  # context
    )

    definitions = [
            {'data-en_def': item['data-en_def']}
        for item in definitions
        ]
    return {"input": input_prompt, "definitions": definitions}

def make_data(out_dir):
    # Load Cambridge Dict
    with open(os.path.join(DATA_FOLDER, FILE_CAM_DICT), "r") as f:
        cambridge_dict = json.load(f)

    examples, non_examples = [], []
    for lem in tqdm(cambridge_dict):
        if lem in STOPWORDS:
            continue
        dict_entry = cambridge_dict[lem]
        
        for pos in dict_entry:
            all_senses = []
            target_sense_num = 1

            for item1 in dict_entry[pos]:
                for item2 in item1["big_sense"]:
                    for sense_block in item2["sense"]:    
                        all_senses.append(
                            {
                                "data-en_def": preprocess_definition(sense_block['en_def']),
                                "target_sense_num": target_sense_num,
                                "example_sents": [ex['en'] for ex in sense_block['examples']]
                            }
                        )
                        target_sense_num += 1
            if len(all_senses) > 1:
                # separate each example sent to be 1 dataset example
                for sense in all_senses:
                    for ex in sense['example_sents']:
                        if not ex:
                            continue
                        if ex[-1] not in string.punctuation:
                            continue
                        formatted_prompt_input = format_prompt_input(lem, ex, all_senses)
                        if formatted_prompt_input is not None:
                            examples.append(
                                {
                                    **formatted_prompt_input, # input, all definitions
                                    "sent": ex,
                                    "lemma": lem,
                                    "target_sense_num": sense['target_sense_num'], # answer
                                }
                            )
                        else:
                            non_examples.append(sense)
            else:
                non_examples.append(lem)
    print(f"{len(examples)} examples saved ; {len(non_examples)} not used!")
    with jsonlines.open(os.path.join(out_dir, "camb.test.jsonl"), "w") as f:
        f.write_all(examples)

if __name__=="__main__":
    
    make_data(sys.argv[1])

    # data = [line for line in jsonlines.open("camb.test.jsonl")]
    
    # random.seed(42)
    # test = random.sample(data, int(len(data)/10))
    # with jsonlines.open("camb.test.sampled.jsonl", "w") as f:
    #     f.write_all(test)
    
