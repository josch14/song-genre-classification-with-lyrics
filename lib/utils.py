
import json
import numpy as np
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Bidirectional, Embedding

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

    return Embedding(input_dim=vocab_len, output_dim=glove_dim, input_length=max_length, weights=[emb_matrix], trainable=True)
