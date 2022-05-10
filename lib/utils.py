
import json
import numpy as np
import tensorflow
import pandas as pd
from constants import DATA_ARTISTS

from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
import contractions
from constants import SYMBOLS

"""
Sort a dictionary by its values.
"""


def sort_dict(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))


"""
Pretty print a dictionary.
"""


def print_dict(d: dict, sorted=False) -> None:
    if sorted:
        d = sort_dict(d)
    print(json.dumps(d, indent=4))


def lyrics_to_verses(lyrics: str) -> list:
    return lyrics.split("\n")


def verses_to_lyrics(verses: list) -> str:
    return "\n".join(verses)


def round_float(number: float, digits: int = 2):
    return round(number, digits)

# """
# Use Glove.
# https://towardsdatascience.com/sentiment-analysis-using-lstm-and-glove-embeddings-99223a87fe8e
# """

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
        return word_to_vec_map


def get_glove_embedding(glove_dim: int, max_length: int, word_to_vec_map: dict, words_to_index: dict):
    vocab_len = len(words_to_index)

    emb_matrix = np.zeros((vocab_len, glove_dim))
    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index-1, :] = embedding_vector

    return tensorflow.keras.layers.Embedding(input_dim=vocab_len, output_dim=glove_dim, input_length=max_length, weights=[emb_matrix], trainable=True)



def get_glove_dim(glove_filename: str):
    if "50d" in glove_filename:
        glove_dim = 50
    elif "100d" in glove_filename:
        glove_dim = 100
    elif "200d" in glove_filename:
        glove_dim = 200
    elif "300d" in glove_filename:
        glove_dim = 300
    else:
        exit("Glove Dimension not valid. Exiting ...")
    return glove_dim


def get_early_stopping_callback(patience_epochs: int):
    callback = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience_epochs,
        mode='auto',
        verbose=1,
        restore_best_weights=True,
    )
    return callback

def get_artists():
    artists = pd.read_csv(DATA_ARTISTS)
    return [str(a) for a in artists['Artist'].unique()]


"""
Preprocessing.
"""
en_stopwords = stopwords.words('english')
def remove_whitespaces(text: str):
    return text.strip()

def remove_stopwords(text: str):
    text = [w for w in text.split(" ") if w not in en_stopwords]
    return remove_whitespaces(' '.join(text))

def remove_contractions(text: str):
    text = contractions.fix(text)
    # exceptions, e.g., "god's", "else's" -> then take the first part of the word
    text = [w.split("'")[0] if "'" in w else w for w in text.split(" ")]
    return remove_whitespaces(' '.join(text))

def remove_symbols(text: str):
    for symbol in SYMBOLS:
        text = text.replace(symbol, " ")
    return remove_whitespaces(text)

def remove_artists(text: str, artists: list):
    for artist in artists:
        artist = artist.lower()
        text = text.replace(artist, " ")
    return remove_whitespaces(text)