# Installing the requirements
- NLTK
- Spacy


# Getting SemCor from GlossBERT's preprocessed dataset
Download preprocessed dataset
-> make empy folders:
Data
	examples
	gold_keys

-> go into the preprocessed dataset folder
-> go into Training_Corpora/SemCor
-> move ....train_token_cls.csv into data/examples
   move ....semcor.gold.key.txt into data/gold_keys
-> go into ../../Evaluation_Datasets
   For each folder, move *.text_token_cls.csv into data/examples ; *.gold.key.txt into data/gold_keys

# Getting SemCor from NLTK
