import os
import sys
import pandas as pd
import argparse
from collections import defaultdict
from tqdm import tqdm
from lib import dataset

# spaCy import
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from constants import DATA_FOLDER, TARGET_GENRES, DATA_ARTISTS, DATA_LYRICS, DATA_PROCESSED

"""
Language detection model to ensure that all lyrics are english. 
The following wrapper is necessary, see:
https://stackoverflow.com/questions/66712753/how-to-use-languagedetector-from-spacy-langdetect-package
"""
def get_lang_detector(nlp, name):
    return LanguageDetector()
nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# data frame columns
LYRIC = "Lyric"
ARTIST = "Artist"
GENRES = "Genres"

"""
Some dataset processing (remove songs with two genres, only use songs with english language).
Saves intermediate data to DATA_PROCESSED.
"""
def process_dataset() -> list:
    if not os.path.exists(DATA_LYRICS) and os.path.exists(DATA_ARTISTS):
        sys.exit('Could not find data files ..')

    lyrics = pd.read_csv(DATA_LYRICS)
    lyrics = lyrics[lyrics['language'] == 'en']
    artists = pd.read_csv(DATA_ARTISTS)

    # merge DataFrames, results has columns:
    # song = [Lyric, Artist, Genres]
    df = lyrics.merge(artists[['Artist', 'Genres', 'Link']],
                      left_on='ALink', right_on='Link', how='inner')
    df = df.drop(columns=['SName', 'ALink', 'SLink', 'language', 'Link'])
    df.dropna(subset=["Lyric", "Artist", "Genres"], inplace=True)

    # remove songs that have two genres
    print(f"# songs: {len(df)}")
    df = df[df['Genres'].apply(lambda x: len(x.split('; ')) == 1)]
    print(f"# songs after removing those with multiple genres: {len(df)}")

    # remove songs that do not contain english language
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        doc = nlp(row['Lyric'])  # use spaCy NLP
        detect_language = doc._.language
        if detect_language['language'] != 'en' or detect_language['score'] < 0.99:
            df.drop(index, inplace=True)
    print(f"# songs after removing those due to language: {len(df)}")

    # save processed dataset
    df.to_csv(DATA_PROCESSED, index=False)


def build_and_save_datasets(df: pd.DataFrame, category_song_limit: int) -> None:
    songs_per_category = defaultdict(int)

    all_lyrics, all_genres = [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        lyric = row[LYRIC]
        genre = row[GENRES]
        if genre in TARGET_GENRES and songs_per_category[genre] < category_song_limit:
            songs_per_category[genre] += 1
            all_lyrics.append(lyric)
            all_genres.append(genre)

    for i in range(1, len(TARGET_GENRES)):
        target_genres = TARGET_GENRES[:i+1]
        print(f"\nBuilding dataset for: {target_genres}")

        lyrics, genres = [], []
        for lyric, genre in zip(all_lyrics, all_genres):
            if genre in target_genres:
                lyrics.append(lyric)
                genres.append(genre)

        train_test_df = pd.DataFrame(
            {dataset.GENRE: genres, dataset.LYRICS: lyrics})

        # save data
        path = os.path.join(
            DATA_FOLDER, f"train_test_{len(target_genres)}_genres.csv")
        train_test_df.to_csv(path, index=False)
        print(f"Saved data to {path}")



parser = argparse.ArgumentParser()
parser.add_argument('-n', '--songs_per_category', default=1000000,
                    type=int, help='Maximum number of songs per category.')
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(DATA_PROCESSED):
        process_dataset()

    df = pd.read_csv(DATA_PROCESSED)
    # shuffle data, so that the songs of same artists don't appear next to each other
    df = df.sample(frac=1)

    build_and_save_datasets(df, args.songs_per_category)