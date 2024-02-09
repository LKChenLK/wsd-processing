mkdir GlossBERT_datasets
unzip GlossBERT_datasets.zip -d GlossBERT_datasets

mkdir -p data/examples data/gold_keys

mv ./GlossBERT_datasets/Training_Corpora/SemCor/semcor_train_token_cls.csv ./data/examples
mv ./GlossBERT_datasets/Training_Corpora/SemCor/semcor.gold.key.txt ./data/gold_keys

eval_datasets=("ALL" "semeval2007" "semeval2013" "semeval2015" "senseval2" "senseval3")

# Loop through the array
for d in "${eval_datasets[@]}"
    do
        mv "./GlossBERT_datasets/Evaluation_Datasets/${d}/${d}_test_token_cls.csv" ./data/examples
        mv "./GlossBERT_datasets/Evaluation_Datasets/${d}/${d}.gold.key.txt" ./data/gold_keys
    done

