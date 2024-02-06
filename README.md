# Installing the requirements
First, create a virtual environment (in any way you like) and activate it.
Then, run
```bash
pip install -r requirements.txt
```

# Getting the SemCore dataset

You may obtain SemCor as a training set in 2 ways. This might be important if the input to your model requires that the target word is marked in its context sentence (e.g. "The " bank " by the river is crowded with people.")

The first way (via GlossBERT's preprocessed dataset) ensures that the word marked within a sentence is correct, since this version of the dataset provides the target word's index in the context sentence. However, you'll need to download the dataset manually first.
NOTE: if you want to evaluate your model on the benchmark, you'll want to download the evaluation dataset anyway :P

The second way (via loading SemCor from nltk directly) infers where the word is in a sentence by matching for the lemma of the target word in the target sentence. This is a more direct and simpler way, and you can mostly find the correct index, but if the same word appears in the context sentence more than once, there is possibility for mistake. 

## Getting SemCor from GlossBERT's preprocessed dataset
Download the preprocessed dataset in csv format from [google drive](https://drive.google.com/file/d/1OA-Ux6N517HrdiTDeGeIZp5xTq74Hucf/view)

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

A jsonlines file is mainly easier to read and examine in a command line environment. The test scripts for evaluation also require the input file to be in a jsonlines format.

## Getting SemCor from NLTK

First, install models and data required.
```bash
python -m nltk.downloader wordnet
python -m nltk.downloader semcor
python -m spacy download en_core_web_sm
```

Next, process the data:
```bash
python process_semcor_nltk.py ./data/
```


