COARSE_TO_CAMB = {
    "ADJ": ["adjective", "adj", "combining form"],
    "ADP": ["preposition", "conjunction"],  # not sure which one i should use...
    "ADV": ["adverb"],
    "AUX": ["auxiliary verb", "modal verb"],
    "CCONJ": ["conjunction"],
    "DET": ["determiner", "predeterminer"],
    "INTJ": ["exclamation", "noun or exclamation"],
    "NOUN": ["noun", "noun "],
    "NUM": ["number", "ordinal number"],
    "PART": ["preposition"],  # ?
    "PRON": ["pronoun", "pronoun plural of"],
    "PROPN": ["noun", "noun "],
    "PUNCT": [],
    "SCONJ": ["conjunction"],
    "SYM": ["symbol"],
    "VERB": [
        "verb"
    ],  # excluding "phrasal verb" here, because cambridge.word.888.json entries
    # with both "verb" and "phrasal verb" tags contains the same content under both POS's
    # Examples: "come-out", "back-off", "dally-with-sb"
    "X": ["suffix", "prefix", "idiom", "abbreviation", ""],
}


# we probably only need to do WSD on content words
# and can ignore POS for function words. But I tentatively mapped them out
# just in case.
CAMB_TO_COARSE = {
    "": "X",
    "abbreviation": "X",  # ?
    "adj": "ADJ",
    "adjective": "ADJ",
    "adverb": "ADV",
    "auxiliary verb": "AUX",
    "combining form": "ADJ",  # see notes below
    "conjunction": "CCONJ",
    "determiner": "DET",
    "exclamation": "INTJ",
    "idiom": "X",  # ?
    "modal verb": "AUX",  # Modal verbs are a type of auxiliary verbs
    "noun": "NOUN",
    "noun ": "NOUN",
    "noun or exclamation": "INTJ",  # see notes below
    "number": "NUM",
    "ordinal number": "NUM",
    "phrasal verb": "VERB",  # ?
    "predeterminer": "DET",  # ??
    "prefix": "X",  # ?
    "preposition": "PART",  # or ADP??
    # According to https://www.teachingenglish.org.uk/professional-development/teachers/knowing-subject/n-p/particle
    # a preposition is a type of particle. But (according to me) it is also a kind of adposition.
    "pronoun": "PRON",
    "pronoun plural of": "PRON",
    "suffix": "X",  # ?
    "symbol": "SYM",
    "verb": "VERB",
}

camb_pos = [
    "",  # filed under 'X'
    "abbreviation",  # filed under 'X'
    "adj",
    "adjective",
    "adverb",
    "auxiliary verb",
    "combining form",  # only word under this is "articularis", see camb dict entry
    "conjunction",  # under both 'CCONJ' and 'SCONJ'
    "determiner",
    "exclamation",
    "idiom",  # filed under 'X'
    "modal verb",
    "noun",
    "noun ",
    "noun or exclamation",  # only word is 'ka-ching'
    "number",
    "ordinal number",
    "phrasal verb",
    "predeterminer",  # don't know what maps to this; under 'DET' for now
    "prefix",  # filed under 'X'
    "preposition",  # filed under both 'ADP' and 'PART'
    "pronoun",
    "pronoun plural of",
    "suffix",  # filed under 'X'
    "symbol",
    "verb",
]

coarse_pos = [
    # SpacyToken.pos_
    # https://universaldependencies.org/u/pos/
    "ADJ",  # adjective
    "ADP",  # adposition, incl. pre- and post-position
    "ADV",  # adverb
    "AUX",  # auxiliary
    "CCONJ",  # coordinating conjunction
    "DET",  # determiner
    "INTJ",  # interjection
    "NOUN",  # noun
    "NUM",  # numeral
    "PART",  # particle
    "PRON",  # pronoun
    "PROPN",  # proper noun
    "PUNCT",  # punctuation
    "SCONJ",  # subordinating conjunction
    "SYM",  # symbol
    "VERB",  # verb
    "X",  # other
]

finegrained_pos = [
    # SpacyToken.tag_
    # https://spacy.io/models/en
    "$",
    "",
    ",",
    "-LRB-",
    "-RRB-",
    ".",
    ":",
    "ADD",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NFP",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "XX",
    "_SP",
    "``",
]
