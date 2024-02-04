# import nltk
# nltk.download('semcor')
# nltk.download('wordnet')
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from typing import List, Dict

import jsonlines
import spacy

model_en = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat', 'custom'])

INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

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
            return None
        except IndexError as ie:
            print(f"Error with word {target}: {ie}")
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



# Use NLTK data directly
semcor_data = []
for sent in tqdm(semcor.tagged_sents(tag="both")):
    sent_str = ' '.join([leaf for tok in sent for leaf in tok.leaves()])
    for tok in sent:
        breakpoint()
        if type(tok.label()) != str and tok.label() is not None:
            syn = tok.label().synset()
            pos = tok[0].label()
            word = ' '.join(tok[0].leaves())
            target_def = syn.definition()
            all_defs = [_syn.definition() for _syn in wn.synsets(tok.label().name())]

            # search index of target_def
            target_sense_num = -1
            for i, d in enumerate(all_defs):
                if d == syn.definition():
                    target_sense_num = i
                    break

            examples = []
            if syn.examples():
                for ex in syn.examples():
                    if word in ex:
                        examples.append(ex)

            syn = syn.name()
            token_info = {
                'word': word,
                'synset': syn,
                'target_def_index': target_def,
                'pos': pos,
                'examples': examples,
                'sent': sent_str
            }
            semcor_data.append(token_info)
    
        #breakpoint()
            # tok: (Lemma('group.n.01.group') (NE (NNP Fulton County Grand Jury)))
            # syn: Synset('group.n.01')
            # pos: NE

with jsonlines.open('semcor_data.jsonl', 'w') as f:
    f.write_all(semcor_data)



