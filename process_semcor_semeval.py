# import nltk
# nltk.download('semcor')
# nltk.download('wordnet')
import os
import csv
import argparse
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from typing import List, Dict

import jsonlines
# import spacy

# model_en = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat', 'custom'])

INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

def mark_target(target_word: str, sent: List, target_idx_start: int, target_idx_end: int) -> str:
    out_sent = [tok for tok in sent]
    out_sent[target_idx_start:target_idx_end] = [f'" {target_word} "']
    out_sent = " ".join(out_sent)
    return out_sent

def format_prompt_input(
        target_word: str, 
        target_idx_start: int, 
        target_idx_end: int, 
        sent: List, 
        definitions: List
        ) -> Dict:
    """Turns token and its dictionary definitions + token's context sentence
    into input prompt for WSD model.
    :param word: word to be disambiguated
    :param sent: context sentence of `word`
    :param definitions: List[Dict]. Output of `get_cambridge_dict_info()`
    :return: Dict[str]
    """

    for i, item in enumerate(definitions):
        # We number the definitions starting with 1 also.
        item["numbered_def"] = f"({i + 1}) {item['data-en_def']}"

    context_sent = mark_target(target_word, sent, target_idx_start, target_idx_end)
    if context_sent is None:
        raise NotImplementedError
    input_prompt = INPUT_TEMPLATE.format(
        target_word,
        ' " , " '.join([item["numbered_def"] for item in definitions][:-1]),
        [item["numbered_def"] for item in definitions][-1],  # "def1, ..., or def4"
        context_sent,  # context
    )

    definitions = [
            {'data-en_def': item['data-en_def']}
        for item in definitions
        ]
    return {"input": input_prompt, "definitions": definitions}


def load_keys(input_file: os.PathLike) -> None:
    data = []
    with open(input_file, "r", encoding="'iso-8859-1'") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip the header
        next(reader)      
        prev_target_id = None
        current_sense_idx = -1
        target_sense_idx = -1 # which sense the target word is in the list of senses
        senses = []
        for el in reader:
            target_id, label, sentence, gloss, target_index_start, target_index_end, sense_key = el
            target_index_start, target_index_end = int(target_index_start), int(target_index_end)
            
            # first line of csv file
            if not prev_target_id:
                prev_target_id = el[0]
                continue
            
            if prev_target_id != el[0]:
                prev_target_id = el[0]
                sentence = sentence.split()
                target_word = " ".join(sentence[target_index_start:target_index_end])
                senses = [{'data-en_def': sense} for sense in senses]
                prompt = format_prompt_input(target_word, target_index_start, target_index_end, sentence, senses)
                assert target_sense_idx != -1
                data.append(
                    {
                        "prompt": prompt,
                        "target_sense_idx": target_sense_idx, # 0-based. this helps us find the sense key of the target word, 
                        # which helps us find the corresponding definition
                        "sense_key": sense_key, # ...just in case
                        "target_id": target_id # this helps us find the gold key in the gold key file
                    }
                )
                senses, target_sense_idx = [], -1

            senses.append(gloss)
            current_sense_idx += 1
            if label:
                target_sense_idx = current_sense_idx
    return data



def load_gold_keys(gold_key_file: os.PathLike) -> Dict[str, str]:
    gold_keys = {}
    with open(gold_key_file, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            tmp = s.split()
            gold_keys[tmp[0]] = tmp[1:]
            s = f.readline().strip()
    return gold_keys

def main(args):
    dataset_name = args.dataset_name
    data_dir = args.data_dir

    if 'semcor' not in dataset_name: # test data
        gold_keys = load_gold_keys(os.path.join(data_dir, 'gold_keys', f'{dataset_name}.gold.key.txt'))
        file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv')
        data = load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv'))
    else: # training data
        file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv')
        data = load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv'))
    
    with jsonlines.open(os.path.join(data_dir, dataset_name+".jsonl"), "w") as f:
        f.write_all(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)

