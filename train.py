import os
import argparse

from transformers import AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import logging, sys
from datetime import datetime
import transformers.utils.logging as hf_logging

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# batch-tokenize inputs
def tokenize_batch(batch):
    """ Input: a batch of your dataset
        Example: { 'text': [['sentence1'], ['setence2'], ...],
                   (Optional)'labels': ['correct_sentence1', 'correct_sentence2', ...] }
    """

    # encode the source sentence, i.e. the grammatically-incorrect sentences
    input_sequences = ["disambiguate:" + line for line in  batch['text']]
    input_encoding = tokenizer(
        input_sequences,
        padding="max_length",
        max_length=MODEL_MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = input_encoding.input_ids, \
                                input_encoding.attention_mask

    # encode the targets, i.e. the definitions
    labels = None
    if 'label' in batch.keys():
        output_sequences = batch['label']
        target_encoding = tokenizer(
            output_sequences,
            padding="max_length",
            max_length=MODEL_MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        print(f"WARNING: not tokenizing labels!!")

    ################################################

    """ Output: a batch of processed dataset
        Example: { 'input_ids': ...,
                   'attention_masks': ...,
                   'label': ... }
    """
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
    #loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


def main(args):

    data_files = {'train': os.path.join(args.dataset_path, 'train.jsonl'),\
                'validation': os.path.join(args.dataset_path, 'dev.jsonl'),\
                #'test': os.path.join(args.dataset_path, 'test.jsonl')
                }
    dataset = load_dataset('json', data_files = data_files)

    train_val_dataset = dataset.map(
        tokenize_batch,    # your processing function
        batched = True # Process in batches so it can be faster
        )
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.model_dir,
        do_eval=True,
        evaluation_strategy='epoch',
        save_steps=1000,
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.train_batch_size*4,
        num_train_epochs = args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type
        #remove_unused_columns=False
    )

    training_args.set_logging(report_to=['wandb']) # https://github.com/huggingface/transformers/issues/22429
    # training_args.set_lr_scheduler('constant', num_epochs=EPOCH) # NEED TO FEED `EPOCH` into it, or num_(train)_epochs will always stay at the default 3!!

    # now give all the information to a trainer
    trainer = Seq2SeqTrainer(
        # set your parameters here
        model = model,
        args = training_args,
        train_dataset = train_val_dataset["train"],
        eval_dataset = train_val_dataset["validation"],
        tokenizer = tokenizer,
        # data_collator = data_collator,
    )

    trainer.train()

    model.save_pretrained(args.model_dir)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True) # output model
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True) # output model
    parser.add_argument("--cont_train_model_dir", type=str, default=None) # input model

    # Training-specific args
    parser.add_argument("--train_batch_size", type=int, default=16) # shrink to 8 if use any of the 'choice' prompts
    parser.add_argument("--max_length", type=int, default=100) # expand to 400 of using 'choice' prompts
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=6)


    # Only GPT-2 changes these
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default='linear')

    args = parser.parse_args()

    # max sent len (split by white space) in training set is 367 (multiple-choice); 72 (generative)
    MODEL_MAX_LEN = args.max_length
    MODEL_NAME = args.pretrained_model
    if args.cont_train_model_dir:
        MODEL_NAME=args.cont_train_model_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_length=MODEL_MAX_LEN)
    
    model = None
    if "t5" in MODEL_NAME:
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, max_length=MODEL_MAX_LEN)
    elif "gpt2" in MODEL_NAME:
        from transformers import AutoModelForCausalLM
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, max_length=MODEL_MAX_LEN)

    os.makedirs(args.model_dir, exist_ok=True)
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%d%m%Y-%H%M%S") # ddmmYY-HMS
    logging.basicConfig(
                    handlers=[
                        logging.FileHandler(f"{args.model_dir}/{__name__}_{dt_string}.log", mode='a'),
                        logging.StreamHandler(sys.stdout)
                    ],
                    #format='%(asctime)s.%(msecs)d [%(levelname)s] %(funcName)s - %(message)s' if DEBUG else ' %(message)s',
                    format='%(funcName)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    hf_logging.set_verbosity_info()

    main(args)