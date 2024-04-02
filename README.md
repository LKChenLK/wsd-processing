# Installing the requirements
1. Create a virtual environment under the root of this directory:
    `python -m venv env`

    This creates a virtual environment name `env`. It is important that the environment name is `env`, as it is hard-coded in the scripts used for training and evaluation.
2. Activate your environment:
    `source env/bin/activate`

3. Install the dependencies: run
    ```bash
    pip install -r requirements.txt
    ```

# [OPTIONAL] Getting the SemCore dataset

You may obtain SemCor as a training set in 2 ways. This might be important if the input to your model requires that the target word is marked in its context sentence (e.g. "The " bank " by the river is crowded with people.")

The first way (via GlossBERT's preprocessed dataset) ensures that the word marked within a sentence is correct, since this version of the dataset provides the target word's index in the context sentence. However, you'll need to download the dataset manually first. The zip file from the download link includes SemCor as the training set, and several semeval and senseval evaluation datasets as the standard evaluation benchmark. You won't need them unless you want to evaluate your model on the standard benchmark.

The second way (via loading SemCor from NLTK directly) infers where the word is in a sentence by matching for the lemma of the target word in the target sentence. This is a more direct and simpler way, and you can mostly find the correct index, but if the same word appears in the context sentence more than once, there is possibility for mistake. 

⚠️ NOTE: the two methods stores the output file at the same place with the same name. If you want to try both and be able to distinguish them, you'll have to modify the code (`process_*.py`) and change the output directory or filename to distinguish them.

## Getting SemCor from GlossBERT's preprocessed dataset
Download the preprocessed dataset in csv format from [google drive](https://drive.google.com/file/d/1OA-Ux6N517HrdiTDeGeIZp5xTq74Hucf/view) and put it in the root of this directory.

For training, we only need files that end with `_train_token_cls.csv`.
For evaluation, we need `*gold.key.txt`.

At the root of this directory:

```bash
sh move_files.sh
```
After you run the above script, your `data` directory should look like this:

```
./data
├── examples
│   ├── ALL_test_token_cls.csv
│   ├── semcor_train_token_cls.csv
│   ├── semeval2007_test_token_cls.csv
│   ├── semeval2013_test_token_cls.csv
│   ├── semeval2015_test_token_cls.csv
│   ├── senseval2_test_token_cls.csv
│   └── senseval3_test_token_cls.csv
└── gold_keys
    ├── ALL.gold.key.txt
    ├── semcor.gold.key.txt
    ├── semeval2007.gold.key.txt
    ├── semeval2013.gold.key.txt
    ├── semeval2015.gold.key.txt
    ├── senseval2.gold.key.txt
    └── senseval3.gold.key.txt
 ```

We no longer need the rest of the files in `GlossBERT_datasets`. You can now safely delete the folder in any way you like.

To get a `.jsonl` file of any of the DATASET_NAME in `["semcor", "semeval2007", "semeval2013", "semeval2015", "senseval2", "senseval3"]`, run:
```bash
python process_semcor_semeval.py \
    --dataset_name DATASET_NAME \ 
    --data_dir ./data
```

A jsonlines file is mainly easier to read and examine in a command line environment. The test scripts for evaluation also require the input file to be in the `jsonl` format.

## Getting SemCor from NLTK

First, install the models and data required.
```bash
python -m nltk.downloader wordnet semcor
python -m spacy download en_core_web_sm
```

Next, process the data:
```bash
python process_semcor_nltk.py ./data/
```

# Generate Cambridge data splits
I'm assuming you already have `cambridge.word.888.json`. Make a copy of it with the same file name in `./data/cambridge`.

Then run
```bash
python gen_cambridge_data.py --prompt_type generative --split
```

This will create a folder `./data/cambridge/generative`, under which you will find `camb.prompts.jsonl`, which is split into `{train,dev,test}.jsonl` under the same directory.

## Types of training data
There are 2 modes / types of data (prompt types) the model can be trained on:
- Multiple choice, where the model is trained to choose from a set of definitions a correct sense for a given context and target word. 
    - `multiple_choice` means the model is trained to produce the index of the predicted definition. All the senses (definitions) filled in the template are numbered, thus you can find the text of the definition with the index. 
    - `generative_choices` means the model is trained to produce the entire definition.
- Generative, where the model is trained to generate the definition given only the context sentence and the target word.
    - Use `generative` for this setting

# Train model(s) with the Cambridge data
The main script for training models is `train.py`. The specific hyper-parameters for training are fed into the model via the command line. 


To reproduce the model used in the jupyter notebook, run
```bash
bash train_templates/t5-small-generative.sh 
```

- `--pretrained_model` is the base model we will fine-tune. The name should be on huggingface.
- `--dataset_path`: path to  `{train,val,test}.jsonl`. Each entry should contain the keys `text` and  `label`.
- `--model_dir` is where the fine-tuned model will be saved.
You can find the definitions for the rest of the options in the huggingface documentation.


[OPTIONAL]: to keep a log of the standard output during training, add `2>&1 | tee <name_of_your_log>.log` to your command. For example,

```bash
bash train_templates/t5-small-generative.sh 2>&1 | tee t5-small-generative-26032024-1.log
```
[End OPTIONAL]

Note that the prompt types are specified by inputting the corresponding datasets via the `dataset_path` argument. To train models with other types of prompts, please first generate corresponding datasets (see [previous section](#generate-cambridge-data-splits)).

This code uses wandb for monitoring the training process. This requires that you create a wandb account and logging in (only once) on your server. If you wish not to use it, comment out line 89 in `train.py`:
```python
training_args.set_logging(report_to=['wandb']) 
```

## To continue training from a checkpoint
Refer to the template `train_templates/t5-small-generative-cont.sh`. Note that the difference between this and the previous template.

- `--cont_train_model_dir` is a model checkpoint directory where the model you want to continue training is stored.
- `--model_dir` is the output directory of the model you're about to train. 
- `--pretrained_model` refers to the base model (e.g. `t5-small` or `gpt2`). This is mainly for loading the tokenizer properly.

Run it the same way:
```bash
bash train_templates/t5-small-generative-cont.sh
```


# Evaluate model
To evaluate, run
```bash
bash eval_templates/t5-small-generative.sh
```

- `--test_path`  expects the path of the test set in jsonlines format, with the key "text" containing the target word and its context sentence.
- `--model_dir` expects the directory to the model checkpoint you trained.
- `--out_path` is where the model predictions will be saved.
- `--prompt_type` specifies the type of evaluation. 


For the mode **"multiple_choice"**, the model is evaluated as follows:
1. Do WSD prediction with the finetuned model
2. Map the output index to the definition and check if it is the same as that of the ground truth.
3. Calculate accuracy (number of correct definitions / total definitions)

For the modes **"generative_choices"** and **"generative"**, the accuracy is calculated by checking if the model-generated answer string is equal to the generated string.

⚠️ NOTE: This is not a rigorous evaluation method at all, and can only be taken with a grain of salt. Ask for guidance and use at your own caution.

# Inference in Jupyter Noteook
1. Make sure you open your jupyter notebook in an environment where `transformers` is installed
2. Select a model, or go with the default model (best model in generative mode so far)
3. Input your test sentence as shown in the notebook and get the results.