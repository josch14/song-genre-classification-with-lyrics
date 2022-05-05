import os
import sys
import pandas as pd
import argparse
from collections import defaultdict
from lib import utils
from tqdm import tqdm

# spaCy import 
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

"""
Example call: 
python build_dataset.py --id test -c Pop Rap -n 100 -s 0.75
"""
# args
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--max_genres', default=10, type=int, help='Number of different genres of the dataset with most genres.')
parser.add_argument('-n', '--songs_per_category', default=100, type=int, help='Maximum number of songs per category.')
parser.add_argument('-s', '--train_split', default=0.75, type=float, help='Size of train split.')
args = parser.parse_args()

# data input
DATA_LYRICS = './data/lyrics-data.csv'
DATA_ARTISTS = './data/artists-data.csv'
DATA_PROCESSED = './data/processed-data.csv'

"""
Some dataset processing (remove songs with two genres, only use songs with english language).
Saves intermediate data to DATA_PROCESSED.
"""
def process_dataset() -> list:
    if not os.path.exists(DATA_LYRICS) and os.path.exists(DATA_ARTISTS):
        sys.exit('Could not find data files ..')

    lyrics = pd.read_csv(DATA_LYRICS)
    lyrics = lyrics[lyrics['language']=='en']
    artists = pd.read_csv(DATA_ARTISTS)

    # merge DataFrames, results has columns:
    # song = [Lyric, Artist, Genres]
    df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
    df = df.drop(columns=['SName','ALink','SLink','language','Link'])
    df.dropna(subset = ["Lyric", "Artist", "Genres"], inplace=True)

    # remove songs that have two genres
    print(f"# songs: {len(df)}")
    df = df[df['Genres'].apply(lambda x: len(x.split('; ')) == 1)]
    print(f"# songs after removing those with multiple genres: {len(df)}")

    # remove songs that do not contain english language
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        doc = nlp(row['Lyric']) # use spaCy NLP
        detect_language = doc._.language
        if detect_language['language'] != 'en' or detect_language['score'] < 0.99:
            df.drop(index, inplace=True)
    print(f"# songs after removing those due to language: {len(df)}")

    # save processed dataset
    df.to_csv(DATA_PROCESSED, index=False)  


# def build_datasets():
#     # get genre distribution (we can be sure that one song has only one genre)
#     genre_distribution = defaultdict(int)
#     for genre in df['Genres']:
#         genre_distribution[genre] += 1
#         # try:
#         #     genres = genre_list.split("; ")
#         #     for genre in genres:
#         #         # exception for "Pop/Rock", as we already have Pop & Rock as categories
#         #         if genre != "Pop/Rock": genre_distribution[genre] += 1
#         # except:
#         #     pass # some weird NaN require this
    
#     print(f"# different genres: {len(genre_distribution)}")
#     genre_distribution = utils.sort_dict(genre_distribution)
#     utils.print_dict(genre_distribution)


#     # # build the datasets now
#     # data_frames = []
#     # for i in range(2, args.max_genres+1):
#     #     target_genres = list(genre_distribution.keys())[:i]
#     #     data_frames.append(build_dataset(target_genres, songs_per_category, train_split))
#     # return data_frames



# def build_dataset(artists: pd.DataFrame, lyrics: pd.DataFrame, target_categories: list, songs_per_category: int, train_split: float) -> pd.DataFrame:
#     print(target_categories)

#     for genre_list in artists['Genres']:
#         try:
#             genres = genre_list.split("; ")
#             for genre in genres:
#                 # exception for "Pop/Rock", as we already have Pop & Rock as categories
#                 if genre != "Pop/Rock": genre_distribution[genre] += 1
#         except:
#             pass # some weird NaN require this
    

#     print()
#     return ""


if __name__ == '__main__':
    if not os.path.exists(DATA_PROCESSED):
        process_dataset()

    df = pd.read_csv(DATA_PROCESSED)
    # build_datasets(df, args.songs_per_category, args.train_split)


"""
# different genres: 79
{
    "Rock": 726,
    "Pop": 590,
    "Rom\u00e2ntico": 562,
    "Gospel/Religioso": 557,
    "Pop/Rock": 409,
    "Hip Hop": 325,
    "Rap": 306,
    "Sertanejo": 297,
    "Indie": 291,
    "MPB": 227,
    "Dance": 227,
    "Trilha Sonora": 222,
    "Heavy Metal": 219,
    "Electronica": 201,
    "Ax\u00e9": 191,
    "Rock Alternativo": 183,
    "Hard Rock": 178,
    "Samba": 176,
    "Funk Carioca": 170,
    "R&B": 163,
    "Forr\u00f3": 160,
    "Funk": 148,
    "Pagode": 147,
    "Country": 145,
    "Black Music": 137,
    "Reggae": 124,
    "Folk": 122,
    "Punk Rock": 112,
    "Soul Music": 105,
    "Hardcore": 105,
    "J-Pop/J-Rock": 98,
    "Infantil": 91,
    "Blues": 88,
    "Cl\u00e1ssico": 76,
    "Instrumental": 68,
    "G\u00f3tico": 64,
    "Tecnopop": 63,
    "House": 54,
    "Jazz": 51,
    "Progressivo": 45,
    "World Music": 44,
    "Bossa Nova": 42,
    "Disco": 41,
    "Emocore": 41,
    "Surf Music": 37,
    "Pop/Punk": 37,
    "P\u00f3s-Punk": 36,
    "Samba Enredo": 36,
    "Regional": 33,
    "Reggaeton": 33,
    "Psicodelia": 32,
    "Rockabilly": 31,
    "Velha Guarda": 31,
    "New Wave": 31,
    "Grunge": 30,
    "K-Pop/K-Rock": 30,
    "Jovem Guarda": 30,
    "Soft Rock": 27,
    "Trip-Hop": 24,
    "New Age": 22,
    "Trap": 21,
    "COLET\u00c2NEA": 19,
    "Trance": 18,
    "Chillout": 17,
    "Ska": 16,
    "Classic Rock": 15,
    "Piano Rock": 11,
    "Power-Pop": 11,
    "Industrial": 11,
    "Metal": 10,
    "Fado": 7,
    "Piseiro": 7,
    "Lo-fi": 7,
    "Post-Rock": 6,
    "Kizomba": 6,
    "Tropical House": 5,
    "M\u00fasicas Ga\u00fachas": 5,
    "Electro Swing": 3,
    "Urban": 1
}
"""

"Black Music", ""
"""
Genres that occur together:
{
    "{'Rap', 'Hip Hop', 'Black Music'}": 5221,
    "{'Rap', 'Hip Hop'}": 5083,
    "{'Country'}": 4774,
    "{'Rock'}": 4672,
    "{'Heavy Metal'}": 4394,
    "{'Indie'}": 4289,
    "{'Rock', 'Indie', 'Rock Alternativo'}": 3223,
    "{'Rock', 'Heavy Metal', 'Hard Rock'}": 2826,
    "{'Heavy Metal', 'Hard Rock', 'Rock'}": 2602,
    "{'Pop'}": 2544,
    "{'Pop', 'Pop/Rock', 'Dance'}": 2334,
    "{'Rap'}": 2012,
    "{'Rock', 'Pop', 'Pop/Rock'}": 1935,
    "{'Pop', 'Pop/Rock', 'Rom\u00e2ntico'}": 1898,
    "{'Hard Rock', 'Heavy Metal', 'Rock'}": 1781,
    "{'Rock', 'Indie'}": 1778,
    "{'Folk', 'Rock', 'Country'}": 1689,
    "{'Rock', 'Hard Rock'}": 1614,
    "{'Soul Music', 'R&B'}": 1570,
    "{'Rock', 'Pop/Rock'}": 1527,
    "{'Pop', 'Pop/Rock'}": 1522,
    "{'Gospel/Religioso'}": 1431,
    "{'Hip Hop', 'R&B'}": 1376,
    "{'Pop/Rock'}": 1374,
    "{'Pop/Rock', 'Rock', 'Gospel/Religioso'}": 1372,
    "{'Jazz'}": 1356,
    "{'Pop', 'Dance'}": 1350,
    "{'Punk Rock'}": 1310,
    "{'Punk Rock', 'Rock', 'Hardcore'}": 1300,
    "{'Reggae'}": 1255,
    "{'Pop', 'Dance', 'Electronica'}": 1251,
    "{'Pop', 'Pop/Rock', 'Trilha Sonora'}": 1148,
    "{'Hip Hop'}": 1112,
    "{'Folk'}": 1078,
    "{'Blues'}": 1063,
    "{'Heavy Metal', 'Hard Rock'}": 1056,
    "{'Soft Rock'}": 1003,
    "{'Heavy Metal', 'Rock', 'G\u00f3tico'}": 1000,
    ...
}
"""