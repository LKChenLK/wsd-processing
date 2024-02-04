import jsonlines

from tqdm import tqdm

from gen_wsd import wsd

test = [line for line in jsonlines.open("camb.test.sampled.jsonl")]
pred_defs, gold_defs = [], []
corrects = {
    "baseline": 0,
    "model": 0
}


for line in tqdm(test):
    pred_def, pred_indices = wsd(line)
    pred = pred_indices[0]
    gold = line["target_sense_num"]-1

    pred_defs.append(pred)
    gold_defs.append(gold)

    if pred==gold:
        corrects["model"] += 1
    baseline = 0

    if baseline==gold:
        corrects['baseline'] += 1

print(f"Accuracy:\n" +
      f"\tModel: {corrects['model']/len(test):.2%}\n" + 
      f"\tBaseline: {corrects['baseline']/len(test):.2%}")


with jsonlines.open("wsd.test.pred.jsonl", "w") as f:
    f.write_all(pred_defs)

