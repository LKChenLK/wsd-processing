import os
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_PATH, "data")

MODEL_NAME = "t5-small"
MODEL_MAX_LEN = 400 # max sent len (split by white space) in training set is 367

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# batch-tokenize inputs
def tokenize_batch(batch):
    """ Input: a batch of your dataset
        Example: { 'text': [['sentence1'], ['setence2'], ...],
                   'corrected': ['correct_sentence1', 'correct_sentence2', ...] }
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

    OUTPUT_DIR = './model'
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 8
    EPOCH = 5
    training_args = Seq2SeqTrainingArguments(
        output_dir = OUTPUT_DIR,
        do_eval=True,
        evaluation_strategy='epoch',
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE*4,
        num_train_epochs = EPOCH,
        #remove_unused_columns=False
        # you can set more parameters here if you want
    )

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

    model.save_pretrained(os.path.join('model', 'finetuned'))

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

    args = parser.parse_args()

    main(args)