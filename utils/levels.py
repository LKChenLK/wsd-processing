import os
import json

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_FOLDER = os.path.join(BASE_PATH, "data")

WORD_POS_PATH = os.path.join(DATA_FOLDER, "lemma_pos.json")
STOPWORDS_PATH = os.path.join(DATA_FOLDER, "stopwords.txt")
AKL_PATH = os.path.join(DATA_FOLDER, "AKL.json")
CEEC_VOC_PATH = os.path.join(DATA_FOLDER, "ceec_vocabulary.json")

AKL = json.load(open(AKL_PATH, "r"))
CEEC_VOC = json.load(open(CEEC_VOC_PATH, "r"))
STOPWORDS = set([line.strip() for line in open(STOPWORDS_PATH)])
WORD_POS = json.load(open(WORD_POS_PATH))


def checkAKL(word, pos):
    if pos in AKL.keys() and word.lower() in AKL[pos]:
        return True
    if word.lower() in AKL["OTHER"]:
        return True
    return False


def check_extra_label(_word, pos):
    label = set()
    word = _word.lower()
    if checkAKL(word, pos):
        label.add("K")
    # check HS level
    if pos in CEEC_VOC.keys():
        for level, voc_list in CEEC_VOC[pos].items():
            if word in voc_list:
                label.add(level)
                if int(level) < 4:
                    label.add("4k")
                else:
                    label.add("7k")
    else:
        for level, voc_list in CEEC_VOC["OTHER"].items():
            if word in voc_list:
                label.add(level)
                if int(level) < 4:
                    label.add("4k")
                else:
                    label.add("7k")
    return list(label)


def cleanup_level(level):
    if level == "" or level == "O":
        return "D"
    else:
        return level
