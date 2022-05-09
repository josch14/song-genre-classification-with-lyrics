# song-lyrics-classification
Genre classification of songs using lyrics.

## Installation

```
conda create -n song_classification python=3.8
conda activate song_classification

conda install -c anaconda pandas 
conda install -c conda-forge tqdm 

conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
pip install spacy-langdetect

conda install -c anaconda tensorflow

conda install -c anaconda scikit-learn 

pip install numpy==1.19.5 # because of some occuring error
```

## Dataset
Download from `https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres` and rename to `./data/`.