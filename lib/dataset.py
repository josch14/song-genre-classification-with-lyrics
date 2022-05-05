import pandas as pd
from collections import defaultdict
from .utils import lyrics_to_verses, verses_to_lyrics
import tensorflow as tf

# data folder
DATA_FOLDER ='./data'

# data frame columns
LYRICS = "Lyrics"
GENRE = "Genre"

from tensorflow.data.experimental import AUTOTUNE

GENRE_2_LABEL = {
    "Pop": 0,
    "Rock": 1,
    "Rap": 2,
    "Country": 3,
    "Reggae": 4,
    "Heavy Metal": 5,
    "Blues": 6,
    "Indie": 7,
    "Hip Hop": 8,
    "Jazz": 9,
    "Folk": 10,
    "Gospel/Religioso": 11}

LABEL_2_GENRE = {
    0: "Pop",
    1: "Rock",
    2: "Rap",
    3: "Country",
    4: "Reggae",
    5: "Heavy Metal",
    6: "Blues",
    7: "Indie",
    8: "Hip Hop",
    9: "Jazz",
    10: "Folk",
    11: "Gospel/Religioso"}

class Dataset:
    def __init__(self, n_target_genres: int, train_split: float = 0.75):
        self.n_target_genres = n_target_genres
        self.x_train, self.x_val, self.y_train, self.y_val = [], [], [], []
        self.__build(train_split)
        self.__processing()


    def __build(self, train_split: float):
        path = f"{DATA_FOLDER}/train_val_{self.n_target_genres}_genres.csv"
        df = pd.read_csv(path)

        # at this point, the data, i.e., the rows in the dataframe, are already shuffled
        # don't shuffle them again, so that we always have the same samples in train and validation,
        # for all datasets

        train_samples_per_category = int((len(df)/self.n_target_genres)*train_split)
        train_songs_per_category = defaultdict(int)

        for index, row in df.iterrows():
            lyrics = row[LYRICS]
            genre = row[GENRE]
            if train_songs_per_category[genre] < train_samples_per_category:
                train_songs_per_category[genre] += 1
                self.x_train.append(lyrics)
                self.y_train.append(genre)
            else:
                self.x_val.append(lyrics)
                self.y_val.append(genre)

    def __processing(self):
        self.x_train = [process_lyrics(x) for x in self.x_train]
        self.x_val = [process_lyrics(x) for x in self.x_val]

    def to_tf_dataset(self, batch_size: int):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, [GENRE_2_LABEL[y] for y in self.y_train]))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, [GENRE_2_LABEL[y] for y in self.y_val]))
        
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.prefetch(1), val_dataset.prefetch(AUTOTUNE)


def process_lyrics(lyrics: str):
    # filter out verses
    verses = lyrics_to_verses(lyrics)

    # 1) [Chorus], [Hook], [E-40], [Chorus:], [Nas], ...
    verses = [v for v in verses if "[" not in v]

    # elements of a song: introduction, verse, pre-chorus, chorus, refrain, post-chorus, bridge, outro, instrumental
    # 2) "(Chorus)", "Chorus", "(Repeat Chorus)", ...
    words = ["introduction", "verse", "pre-chorus", "chorus", "refrain", "post-chorus", "bridge", "outro", "instrumental", "repear"]
    for word in words:
        verses = [v for v in verses if not (word in v.lower() and len(v) < len(word)+6)]

    # 3) weird occurence of many verses like: "MercyMe - Spoken For Lyrics"
    verses = [v for v in verses if not "MercyMe" in v]

    return verses_to_lyrics(verses)
