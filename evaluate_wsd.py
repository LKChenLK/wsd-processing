import os
import jsonlines
from datasets import load_dataset
from datasets import Features, Value
import argparse
from tqdm import tqdm
from datetime import datetime

from train import tokenize_batch

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def evaluate_generative(args):
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Text2TextGenerationPipeline,
    )
    MODEL_NAME = os.path.join(BASE_PATH, args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=50, device='cuda:0')

    #test = [line for line in jsonlines.open(args.test_path)]
    
    corrects = {
        "model": 0,
        "baseline": None
    }

    data_files = {'test': args.test_path}
    ft = Features({'text': Value('string')})
    dataset = load_dataset('json', data_files = data_files, features=ft) # only load the 'text' column

    dataset = dataset.map(
        tokenize_batch,    # your processing function
        batched = True # Process in batches so it can be faster
        )
    
    print('Prediction...')
    start = datetime.now()
    test = pipe(dataset['test']['text'], batch_size=64, clean_up_tokenization_spaces=True)
    print(f"Done predicting in {datetime.now()-start}s. Start Evaluating...")
    pred_defs = []
    gold_defs = [line['label'] for line in jsonlines.open(args.test_path)]
    for i, pred_def in tqdm(enumerate(test)):
        #pred_def = pipe(line['text'])
        gold = gold_defs[i]

        pred_defs.append(pred_def)
        gold_defs.append(gold)

        if pred_def['generated_text']==gold:
            corrects["model"] += 1
            
    res = f"Accuracy:\n" + \
    f"\tModel: {corrects['model']/len(test):.2%}\n"
    
    print(res)

    res_path = "".join(args.out_path.split('.')[:-1])+".txt"
    with open(res_path, "w") as f:
        f.write(res)


    with jsonlines.open(args.out_path, "w") as f:
        f.write_all(pred_defs)


def evaluate_multiple_choice(args):
    from gen_wsd import wsd
    in_file, out_file = args.test_path, args.out_path
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
        evaluate_generative(args)
    elif args.prompt_type=="multiple_choice":
        evaluate_multiple_choice(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True) # data/cambridge/test.jsonl
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--prompt_type", choices=['generative', 'multiple_choice'], required=True)

    args = parser.parse_args()

    main(args)
