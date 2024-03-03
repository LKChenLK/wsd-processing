import os
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import logging, sys
from datetime import datetime
import transformers.utils.logging as hf_logging


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

    # encode the targets, i.e. the corrected sentences
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

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    EPOCH = 5
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.model_dir,
        do_eval=True,
        evaluation_strategy='epoch',
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE*4,
        num_train_epochs = EPOCH,
        #remove_unused_columns=False
    )
    training_args.set_logging(report_to=['wandb']) # https://github.com/huggingface/transformers/issues/22429
    training_args.set_lr_scheduler('constant')

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
    # input_ids = tokenizer(
    #     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    # ).input_ids  # Batch size 1
    # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

    # forward pass
    # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    # last_hidden_states = outputs.last_hidden_state
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="generative")

    args = parser.parse_args()

    BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    DATA_FOLDER = os.path.join(BASE_PATH, "data")

    # max sent len (split by white space) in training set is 367 (multiple-choice); 72 (generative)
    MODEL_MAX_LEN = 400 if "choice" in args.prompt_type else 100
    MODEL_NAME = "t5-small"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, max_length=MODEL_MAX_LEN)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, max_length=MODEL_MAX_LEN)

    
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