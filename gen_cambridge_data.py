#  This is the cleaned version of `gen_camb_test.py` for sharing

#import pdb, traceback
import logging
import os, string
import json, jsonlines
import argparse
import spacy
from tqdm import tqdm
from typing import Dict, List
from sklearn.model_selection import train_test_split
from utils.sense_dicts import preprocess_definition

logging.basicConfig(filename="gen_cambridge_data.log")
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# DATA_FOLDER = "/home/nlplab/kedy/NLP/AES/data"
FILE_CAM_DICT = "cambridge.word.888.json"
STOPWORDS_PATH = os.path.join(BASE_PATH, "data", "stopwords.txt")
STOPWORDS = set([line.strip() for line in open(STOPWORDS_PATH)])

model_en = spacy.load('en_core_web_md', disable=['parser', 'ner', 'textcat', 'custom'])


def format_prompt_input(word: str, sent: str, definitions: List) -> Dict:
    """Turns token and its dictionary definitions + token's context sentence
    into input prompt for WSD model.
    :param word: word to be disambiguated
    :param sent: context sentence of `word`
    :param definitions: List[Dict]. Output of `get_cambridge_dict_info()` in linggle booster's word_info
    :return: Dict[str]
    """
    def mark_target(sent: str, target: str) -> str:
        sent = model_en(sent)
        sent_info = {
                word.lemma_.lower(): {"idx": word.i, "text": word.text,}
                    for word in sent
                }
        try: # unlemmatised target word
            lemma = target.lower()
            target_idx = sent_info[lemma]["idx"] # target.lower() should be the lemma
        except KeyError as ke:
            try: # lemmatise the target word too
                target_words = model_en(target)
                lemma = " ".join([w.lemma_ for w in target_words])
                target_idx = sent_info[lemma]["idx"]
            # Add another try-except:
            # tokenize by white space and make sent_info again 
            #   => don't lemmatize if "-" exists in target
            #   => lemmatize + lower both context words and target word otherwise
            except:
                logging.warning(f"Key error with word '{target}': {sent_info.keys()}")
                #breakpoint()
                return None
        except IndexError as ie:
            logging.warning(f"Index error with word '{target}': {ie}")
            return None
        target_orig_form = sent_info[lemma]['text']
        out_sent = [tok.text for tok in sent]
        out_sent[target_idx] = f'" {target_orig_form} "'
        out_sent = " ".join(out_sent)
        return out_sent

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

def make_data(camb_dir):
    # Load Cambridge Dict
    with open(os.path.join(camb_dir, FILE_CAM_DICT), "r") as f:
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

            # separate each example sent to be 1 dataset example
            for sense in all_senses:
                for ex in sense['example_sents']:
                    if not ex:
                        continue
                    if ex[-1] not in string.punctuation:
                        # only include full sentences as examples
                        continue
                    if "-" in lem:
                        non_examples.append(sense)
                        continue
                    formatted_prompt_input = format_prompt_input(lem, ex, all_senses)
                    if formatted_prompt_input is not None:
                        examples.append(
                            {
                                **formatted_prompt_input, # input, all definitions
                                "sent": ex,
                                "lemma": lem,
                                "target_sense_num": sense['target_sense_num'], # answer
                                "data-en_def": sense["data-en_def"], 
                            }
                        )
                    else:
                        non_examples.append(sense)
    
   
    out_path = os.path.join(camb_dir, "camb.prompts.jsonl")
    with jsonlines.open(out_path, "w") as f:
        f.write_all(examples)
    logging.info(f"{len(examples)} examples of the entire {FILE_CAM_DICT} saved to {out_path}; {len(non_examples)} not used!")
    

def split_dataset(dataset_path, prompt_type):

    def format_out(x, y):
        out_data = []
        for x_, y_ in zip(x, y):
            out_data.append({"text": x_, "label": y_})
        return out_data
    
    data = [line for line in jsonlines.open(os.path.join(dataset_path, "camb.prompts.jsonl"))]
    x_all = [it['input'] for it in data]
    y_all = []
    if prompt_type == "generative":
        y_all = [it['data-en_def'] for it in data]
    elif prompt_type=="multiple_choice":
        y_all = [it['target_sense_num'] for it in data]

    x_train, x_test, y_train, y_test \
        = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

    x_test, x_dev, y_test, y_dev \
        = train_test_split(x_test, y_test, test_size=0.5, random_state=42) # 0.2 * 0.5 = 0.1
    # We end up with a 80, 10, 10 split

    train_out = format_out(x_train, y_train)
    dev_out = format_out(x_dev, y_dev)
    test_out = format_out(x_test, y_test)

    with jsonlines.open(os.path.join(dataset_path, "train.jsonl"), "w") as f:
        f.write_all(train_out)

    with jsonlines.open(os.path.join(dataset_path, "dev.jsonl"), "w") as f:
        f.write_all(dev_out)

    with jsonlines.open(os.path.join(dataset_path, "test.jsonl"), "w") as f:
        f.write_all(test_out)

    logging.info(f"Split dataset saved to {dataset_path}/{{train,dev,test}}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cambridge_dict_dir", default=f"{BASE_PATH}/data/cambridge")
    parser.add_argument("--prompt_type", choices=['generative', 'multiple_choice'], required=True)
    parser.add_argument("--split", action='store_true')
    
    args = parser.parse_args()
    dataset_path = os.path.join(args.cambridge_dict_dir, args.prompt_type)
    os.makedirs(dataset_path, exist_ok=True)
    make_data(dataset_path)
    if args.split:
        split_dataset(dataset_path, args.prompt_type)

if __name__=="__main__":
    # try:
    #     main()
    # except:
    #     pass
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
    main()