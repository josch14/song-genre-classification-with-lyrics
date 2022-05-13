import random
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import numpy as np
import math

# local imports
from constants import GENRE_2_LABEL
from .utils import lyrics_to_verses, verses_to_lyrics

# data folder
DATA_FOLDER = './data'

# data frame columns
LYRICS = "Lyrics"
GENRE = "Genre"


class Dataset:
    def __init__(self, n_target_genres: int, train_split: float = 0.7):
        self.n_target_genres = n_target_genres
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        self.__build(train_split)
        self.__processing()

    def __build(self, train_split: float):
        path = f"{DATA_FOLDER}/train_test_{self.n_target_genres}_genres.csv"
        df = pd.read_csv(path)

        all_lyrics, all_genres = [], []
        for index, row in df.iterrows():
            all_lyrics.append(row[LYRICS])
            all_genres.append(row[GENRE])

        self.x_train, self.y_train, self.x_test, self.y_test = get_splits(
            all_lyrics, all_genres, train_split)

        # shuffle train set
        # (not before, so that any dataset (i.e., with more genres) still has the same songs in train and test set)
        shuffle_list = list(zip(self.x_train, self.y_train))
        random.shuffle(shuffle_list)
        self.x_train, self.y_train = zip(*shuffle_list)
        self.x_train = list(self.x_train)
        # werid error occured that y_train was suddenly a tuple, therefore cast
        self.y_train = list(self.y_train)

    def __processing(self):
        self.x_train = [process_lyrics(x) for x in self.x_train]
        self.x_test = [process_lyrics(x) for x in self.x_test]

    def to_tf_dataset(self, batch_size: int):
        train_set = tf.data.Dataset.from_tensor_slices(
            (self.x_train, [GENRE_2_LABEL[y] for y in self.y_train]))
        test_set = tf.data.Dataset.from_tensor_slices(
            (self.x_test, [GENRE_2_LABEL[y] for y in self.y_test]))

        train_set = train_set.shuffle(1000).batch(batch_size)
        test_set = test_set.shuffle(1000).batch(batch_size)

        return train_set.prefetch(AUTOTUNE), test_set.prefetch(AUTOTUNE)

    def to_numpy_dataset(self):
        return np.array(self.x_train), np.array([GENRE_2_LABEL[y] for y in self.y_train]), np.array(self.x_test), np.array([GENRE_2_LABEL[y] for y in self.y_test])


"""
Make train and test split evenly balanced, i.e., make sure they got the same genre distribution.
"""


def get_splits(all_lyrics: list, all_genres: list, train_split: float):

    # how many songs per genre?
    songs_per_genre = defaultdict(int)
    for genre in all_genres:
        songs_per_genre[genre] += 1

    x_train, y_train, x_test, y_test = [], [], [], []

    for target_genre, n_songs in songs_per_genre.items():
        # Look at each genre separately
        n_songs_train = math.floor(n_songs*train_split)

        n_train, n_test = 0, 0
        for lyrics, genre in zip(all_lyrics, all_genres):
            if genre == target_genre:
                if n_train < n_songs_train:
                    x_train.append(lyrics)
                    y_train.append(genre)
                    n_train += 1
                else:
                    x_test.append(lyrics)
                    y_test.append(genre)
                    n_test += 1

        # print(f"Category: {target_genre}: {n_train}/{n_test}, total: {n_songs}")

    return x_train, y_train, x_test, y_test


"""
Some minor lyrics processing.
"""


def process_lyrics(lyrics: str):
    # filter out verses
    verses = lyrics_to_verses(lyrics)

    # 1) [Chorus], [Hook], [E-40], [Chorus:], [Nas], ...
    verses = [v for v in verses if "[" not in v]

    # elements of a song: introduction, verse, pre-chorus, chorus, refrain, post-chorus, bridge, outro, instrumental
    # 2) "(Chorus)", "Chorus", "(Repeat Chorus)", ...
    words = ["introduction", "verse", "pre-chorus", "chorus", "refrain",
             "post-chorus", "bridge", "outro", "instrumental", "repear"]
    for word in words:
        verses = [v for v in verses if not (
            word in v.lower() and len(v) < len(word)+6)]

    # 3) weird occurence of many verses like: "MercyMe - Spoken For Lyrics"
    verses = [v for v in verses if not "MercyMe" in v]

    return verses_to_lyrics(verses)
