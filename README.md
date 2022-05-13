# Lyrics-Based Song Genre Classification
Lyrics-Based Song Genre Classification. Project as part of the lecture TDT4310: Intelligent Text Analytics and Language Understanding at NTNU, 2022.

## Abstract
Musical genres are essential for organizing songs into musical collections and providing well-functioning music recommendation and retrieval. In order to support these methods, songs need to be tagged with their appropriate genre(s). Annotation of genres by humans is time-consuming and costly, while reliable automatic song genre classification is difficult, especially because the boundaries between musical genres are not clearly defined. Thus, song genre classification remains a challenging topic. To this end, we target this task by only using song lyrics. For this, we implement both, traditional and machine learning text classification methods. Furthermore, we investigate how the classification performance of all methods depends on the number of considered genres in the dataset. Our experiments show that the classification performance of text classification methods degrades for increasing number of considered genres. The best results were consistently achieved using the Bernoulli Naive Bayes classifier.

## Installation
```
conda create -n song_classification python=3.8
conda activate song_classification

conda install -c anaconda pandas 
conda install -c conda-forge tqdm 

# ensure only english songs in the dataset
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
pip install spacy-langdetect

# model implementation
conda install -c anaconda tensorflow
conda install -c anaconda scikit-learn 
pip install more-itertools

# text preprocessing
conda install -c anaconda nltk 
pip install contractions

# plots
pip install matplotlib
conda install -c anaconda seaborn 

# fix occuring errors
pip install numpy==1.19.5
python -m pip install PyQt5
```

## Dataset
This repository already contains the processed dataset used in this work. These can be generated by downloading the dataset from https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres, and putting the two files into `./data/`. Afterwards, run 
```
python build_datasets.py
```

Some dataset analysis can be done using 
```
python analyze_dataset.py
```

Also make sure that the `./data/` folder contains the GloVe word vectors (`glove.6B.50d.txt`) and (`glove.6B.100d.txt`), available at https://nlp.stanford.edu/projects/glove/.


## Model Training
The six different methods can be trained using the following commands. These will run training on all 11 datasets, containing 2 up to 12 genres, and save the logs and achieved metric performances directly to a text file. Note that no model weights are saved to disk at any time.

```
python train.py -m naive_bayes_bernoulli > results/results_naive_bayes_bernoulli.txt
python train.py -m naive_bayes_multinomial > results/results_naive_bayes_multinomial.txt
python train.py -m svm > results/results_svm.txt
python train.py -m mlp_glove -lr 1e-3 > results/results_mlp_glove_1e-3.txt
python train.py -m lstm_glove -lr 1e-3 > results/results_lstm_glove_1e-3.txt
python train.py -m lstm -lr 1e-3 > results/results_lstm_1e-3.txt
```

## Generate metric plots and tables
The achieved metric performances from training were manually written to a json file (see `./evaluation/results_overview.json`). Based on this, the following command generates metric plots prints performance tables for LaTeX.
```
python show_results.py
```

## Detailed evaluation
The following commands allow a more detailed model evaluation (generate metrics per musical genre, and generate confusion matrix):
```
python evaluate_model.py -m naive_bayes_bernoulli
python evaluate_model.py -m naive_bayes_multinomial
python evaluate_model.py -m svm 
python evaluate_model.py -m mlp_glove -lr 1e-3
python evaluate_model.py -m lstm_glove -lr 1e-3
python evaluate_model.py -m lstm -lr 1e-3
```


