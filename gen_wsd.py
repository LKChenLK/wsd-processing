# Modified from Yi-Chen Chang's AESbackend
# https://github.com/NTHU-NLPLAB/AESbackend/blob/main/aes_tools/feedback/vocab/level_analyzer.py
import os
import re
from typing import Dict, List

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Text2TextGenerationPipeline,
)
from .utils.levels import check_extra_label
from .utils.sense_dicts import get_dict_data
from .utils.pos import COARSE_TO_CAMB

BATCH_SIZE = 10

INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_PATH, "data")
# DATA_FOLDER = "/home/nlplab/kedy/NLP/AES/data"
STOPWORDS_PATH = os.path.join(DATA_FOLDER, "stopwords.txt")

# Load stopwords
STOPWORDS = set([line.strip() for line in open(STOPWORDS_PATH)])

"""
Model Info
* T5-small
* SemCor in NLTK Corpus
"""
MODEL_NAME = os.path.join(BASE_PATH, "models/baseline-1-t5-small-opt/")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)


def format_prompt_input(word: str, sent: str, definitions: List) -> Dict:
    """Turns token and its dictionary definitions + token's context sentence
    into input prompt for WSD model.
    :param word: word to be disambiguated
    :param sent: context sentence of `word`
    :param definitions: List[Dict]. Output of `utils.sense_dicts.get_cambridge_dict_data()`
    :return: Dict[str]
    """

    def mark_target(sent: str, target: str) -> str:
        if sent.startswith(f"{target} "):  # replace 1st occurrence only
            new_sent = sent.replace(f"{target} ", f' " {target} " ', 1)
        elif f" {target} " in sent:
            new_sent = sent.replace(f" {target} ", f' " {target} " ')
        return new_sent

    for i, item in enumerate(definitions):
        # Probably because the model was fine-tuned using definition indices
        # that start with 1, we number the definitions starting with 1 also.
        item["numbered_def"] = f"({i + 1}) {item['data-en_def']}"

    input_prompt = INPUT_TEMPLATE.format(
        word,
        ' " , " '.join([item["numbered_def"] for item in definitions][:-1]),
        [item["numbered_def"] for item in definitions][-1],  # "def1, ..., or def4"
        mark_target(sent, word),  # context
    )

    return {"input": input_prompt, "definitions": definitions}


def wsd(token_info: Dict) -> Dict:
    """
    Do word sense disambiguation with generative model, return list of
    disambiguated senses
    :param token_info: Dict containing
      "input": Prompt for the WSD model. `str` input formatted with
               `INPUT_TEMPLATE`.
      "definitions": List[Dict], with each Dict containing
        "data-level": str
        "data-en_def": str
        "data-ch_def": str
        "numbered_def": str
    :return: candidate_defs: List[Dict], token_info["definitions"]
        re-ordered by disambiguated (most-likely) sense
    """
    model_input = token_info["input"]
    definitions = token_info["definitions"]

    # If optimising for speed, use len(inputs) or a constant (e.g. 3)
    # if optimising for accuracy (? not verified), use len(inputs)+constant (3?)
    candidate_defs = []
    NUM_OUT_DEFS = len(definitions) + 3

    candidate_indices = []
    for i, model_output_dict in enumerate(
        pipe(
            model_input,
            max_length=5,
            top_k=NUM_OUT_DEFS,
            num_return_sequences=NUM_OUT_DEFS,
            num_beams=NUM_OUT_DEFS,
        )
    ):
        generated_idx = model_output_dict["generated_text"]
        try:
            # get indices in generated text
            # minus one because we're retrieving the definition by index from the 0-based candidate list
            idx = (int(re.search("(\d+)", generated_idx).group(1)) - 1)  
            if idx not in candidate_indices:
                candidate_defs.append(definitions[idx])  # get text of definition by idx from generated text
                candidate_indices.append(idx)
        except Exception as e:
            if i < 2:
                print(
                    {
                        "error_msg": e,
                        "index": generated_idx,
                        "idx": idx,
                        "defs": definitions,
                    }
                )
    return candidate_defs, candidate_indices


def gen_wsd_data(data, dict_name="Cambridge") -> List:
    """Performs word-sense disambuation for each token in the input data
    :param data: WSDRequest object (see main.py); represents a tokenized
    sentence. The Request object is the output of `preprocess_text()` in
    `../preprocess_en/preprocess.py`, which uses spacy for tokenisation,
    lemmatisation, and pos-tagging
    Each item in `data` is a `token`:
                    Dict[str] of spacy token info.
                    Dict ('word': SpacyToken.text,
                        'lemma': SpacyToken.lemma_,
                        'pos': process_en_pos(SpacyToken.tag_),
                        'raw_pos': SpacyToken.pos_,
                        'end': SpacyToken.whitespace_)
    """
    ret = []
    sent = " ".join([token["word"] for token in data])
    for idx, token in enumerate(data):
        word, lemma, pos = token["word"], token["lemma"].lower(), token["raw_pos"]
        # token["raw_pos"] is SpaCy's coarse POS tags, i.e. SpacyToken.pos_
        # For details, see preprocess_text.preprocess

        if lemma in STOPWORDS:
            ret.append([idx, gen_empty_sense()])
            continue

        # Get dict data
        pos = COARSE_TO_CAMB[pos]
        definitions = []
        for p in pos:  # SpaCy POS sometimes corresponds to multiple camb POS's
            pos_defs = get_dict_data(lemma, p, dict_name=dict_name)
            if pos_defs:
                definitions.extend(pos_defs)

        # Do WSD or return directly
        if not definitions:
            ret.append([idx, gen_empty_sense()])
        elif len(definitions) == 1:
            ret_defs = definitions[0]
            ret_defs["data-label"] = check_extra_label(word, token["pos"])
            ret_defs["data-extra_sense"] = ""
            ret.append([idx, ret_defs])
        else:
            token_info = format_prompt_input(word, sent, definitions)
            candidate_defs, candidate_indices = wsd(token_info)
            ret.append([idx, gen_multisense_data(data[idx], candidate_defs)])
    return ret


def gen_multisense_data(word: Dict, candidate_defs: List) -> Dict:
    """Take most likely sense from WSD, the rest goes to "data-extra_sense".
    :param word: stores token info: (
                    'word': SpacyToken.text
                    'lemma': SpacyToken.lemma_,
                    'pos': process_en_pos(SpacyToken.tag_),
                    'raw_pos': SpacyToken.pos_,
                    'end': SpacyToken.whitespace_)
                    )
    :param candidate_defs: List[Dict], each Dict is a sense from the source dict
    (see: `utils.sense_dicts.get_cambridge_dict_data()`)
    """
    extra_labels = check_extra_label(word["lemma"], word["pos"])
    sense_info = {
        "data-en_def": candidate_defs[0]["data-en_def"],
        "data-ch_def": candidate_defs[0]["data-ch_def"],
        "data-level": candidate_defs[0]["data-level"],
        "data-label": extra_labels,
        "data-extra_sense": " / ".join(
            sense["data-ch_def"] for sense in candidate_defs[1:]
        ),
    }

    return sense_info


def gen_empty_sense() -> Dict:
    sense_info = {}
    sense_info["data-level"] = ""
    sense_info["data-label"] = ""
    sense_info["data-en_def"] = ""
    sense_info["data-ch_def"] = ""
    sense_info["data-extra_sense"] = ""

    return sense_info


if __name__ == "__main__":
    # fmt: off
    # This is an example input data
    in_data = [
        {
            "word": "he",
            "lemma": "he",
            "pos": "PRON",
            "raw_pos": "PRON",
            "end": " "
        },
        {
            "word": "composed",
            "lemma": "compose",
            "pos": "V",
            "raw_pos": "VERB",
            "end": " ",
        },
        {
            "word": "the",
            "lemma": "the",
            "pos": "DT",
            "raw_pos": "DET",
            "end": " "
        },
        {
            "word": "First",
            "lemma": "First",
            "pos": "N",
            "raw_pos": "PROPN",
            "end": " "
        },
        {
            "word": "Violin",
            "lemma": "Violin",
            "pos": "N",
            "raw_pos": "PROPN",
            "end": " ",
        },
        {
            "word": "Sonata",
            "lemma": "Sonata",
            "pos": "N",
            "raw_pos": "PROPN",
            "end": " ",
        },
        {
            "word": "four",
            "lemma": "four",
            "pos": "CD",
            "raw_pos": "NUM",
            "end": " "
        },
        {
            "word": "years",
            "lemma": "year",
            "pos": "N",
            "raw_pos": "NOUN",
            "end": " "
        },
        {
            "word": "earlier",
            "lemma": "early",
            "pos": "ADV",
            "raw_pos": "ADV",
            "end": "",
        },
        {
            "word": ".",
            "lemma": ".",
            "pos": ".",
            "raw_pos": "PUNCT",
            "end": ""
        },
    ]
    # fmt: on
    res = gen_wsd_data(in_data)  # main function of this api
