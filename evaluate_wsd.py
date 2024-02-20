import os, sys
import jsonlines
import argparse
from tqdm import tqdm

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def evaluate_generative(args):
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Text2TextGenerationPipeline,
    )
    MODEL_NAME = os.path.join(BASE_PATH, args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

    test = [line for line in jsonlines.open(args.data_path)]
    pred_defs, gold_defs = [], []
    corrects = {
        "model": 0,
        "baseline": None
    }

    for line in tqdm(test):
        pred_def = pipe(test['text'])
        gold = line["label"]

        pred_defs.append(pred_def)
        gold_defs.append(gold)

        if pred_def==gold:
            corrects["model"] += 1
            
    res = f"Accuracy:\n" + \
    f"\tModel: {corrects['model']/len(test):.2%}\n"
    
    print(res)

    res_path = "".join(args.out_path.split('.')[:-1])+".txt"
    with open(res_path, "w") as f:
        f.write(res)


    with jsonlines.open(out_file, "w") as f:
        f.write_all(pred_defs)


def evaluate_multiple_choice(args):
    from gen_wsd import wsd
    in_file, out_file = args.data_path, args.out_path
    test = [line for line in jsonlines.open(in_file)]
    pred_defs, gold_defs = [], []
    corrects = {
        "baseline": 0,
        "model": 0
    }
    # TODO: modify `wsd()` in `gen_wsd.py` s.t. model path is customisable!!

    out_defs = []
    baseline = 0 # 1st definition
    for line in tqdm(test):
        pred_defs, pred_indices = wsd(line)
        pred = pred_indices[0]
        gold = line["target_sense_num"]-1

        out_defs.append(pred)
        gold_defs.append(gold)

        if pred==gold:
            corrects["model"] += 1
        

        if baseline==gold:
            corrects['baseline'] += 1

    res = f"Accuracy:\n" + \
    f"\tModel: {corrects['model']/len(test):.2%}\n" + \
    f"\tBaseline: {corrects['baseline']/len(test):.2%}"
    
    print(res)

    with open(out_file+".txt", "w") as f:
        f.write(res)


    with jsonlines.open(out_file, "w") as f:
        f.write_all(pred_defs)

def main(args):
    if args.prompt_type=="generative":
        raise NotImplementedError
    elif args.prompt_type=="multiple_choice":
        evaluate_multiple_choice(args.test_path, args.out_path)

        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True) # data/cambridge/test.jsonl
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--prompt_type", choices=['generative', 'multiple_choice'], required=True)

    args = parser.parse_args()

    main(args)
