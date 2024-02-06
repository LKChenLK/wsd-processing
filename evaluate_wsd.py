import sys
import jsonlines
from tqdm import tqdm
from .gen_wsd import wsd



def evaluate(in_file, out_file):
    test = [line for line in jsonlines.open(in_file)]
    pred_defs, gold_defs = [], []
    corrects = {
        "baseline": 0,
        "model": 0
    }

    out_defs = []

    for line in tqdm(test):
        pred_defs, pred_indices = wsd(line)
        pred = pred_indices[0]
        gold = line["target_sense_num"]-1

        out_defs.append(pred)
        gold_defs.append(gold)

        if pred==gold:
            corrects["model"] += 1
        baseline = 0

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

if __name__=="__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    evaluate(in_file, out_file)
