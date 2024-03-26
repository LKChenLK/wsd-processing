import os, json
from typing import List

from utils.levels import cleanup_level

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(BASE_PATH, "data")
# DATA_FOLDER = "/home/nlplab/kedy/NLP/AES/data"
FILE_CAM_DICT = "cambridge.word.888.json"

# Load Cambridge Dict
with open(os.path.join(DATA_FOLDER, "cambridge", FILE_CAM_DICT), "r") as f:
    cambridge_dict = json.load(f)


def preprocess_definition(text: str) -> str:
    text = text.replace("(", "").replace(")", "")
    return text.lower()


def get_dict_data(lemma: str, pos: str, dict_name: str = None) -> List:
    if dict_name == "Cambridge":
        return get_cambridge_dict_data(lemma, pos)
    else:
        raise NotImplementedError


def get_cambridge_dict_data(lemma: str, pos: str) -> List:
    """
    Gets sense, definition, and level info from cambridge dictionary data
    :param pos: part-of-speech. Should be converted to one of Cambridge
    Dictionary's parts of speech. (See utils.pos.camb_pos)
    """

    def get_pos_senses(_pos: str, _dict_entry: dict):
        """Returns a List[Dict] of definitions given part-of-speech and
        dictionary entry of a lemma
        """
        _pos_definitions = []
        for item1 in _dict_entry[_pos]:
            for item2 in item1["big_sense"]:
                for sense_block in item2["sense"]:
                    _pos_definitions.append(
                        {
                            "data-en_def": preprocess_definition(sense_block["en_def"]),
                            "data-ch_def": sense_block["ch_def"],
                            "data-level": cleanup_level(sense_block.get("level", "")),
                        }
                    )
        return _pos_definitions

    dict_entry = cambridge_dict.get(lemma, None)
    if dict_entry:
        definitions = []
        pos_entries = dict_entry.get(pos, None)
        if pos_entries is not None:
            definitions.extend(get_pos_senses(pos, dict_entry))
        # else: # use all POS under the entry
        #     pos_list = list(dict_entry.keys())
        #     if len(pos_list):
        #         for pos_ in pos_list:
        #             definitions.extend(get_pos_senses(pos_, dict_entry))
        return definitions
