from lib.dataset import Dataset
from collections import defaultdict
from lib.utils import print_dict, lyrics_to_verses, sort_dict
from constants import TARGET_GENRES

def main():
    dataset = Dataset(n_target_genres=12)

    x = dataset.x_train + dataset.x_test
    y = dataset.y_train + dataset.y_test

    print(f"Dataset Split: {len(dataset.x_train)}/{len(dataset.x_test)}, in total: {len(x)}")
    # analyze genre distribution
    print(f"Used Genres (in order): {TARGET_GENRES}")
    genre_distribution = analyze_genre_distribution(y)

    # analyze how often tokens occur
    tokens_per_song = analyze_token_numbers(x, y, genre_distribution)

    # print information for LaTeX
    print("Latex Print:")
    cumulative = 0
    for i, genre in enumerate(TARGET_GENRES):
        cumulative += genre_distribution[genre]
        print(f"{i+1} & {genre} & {genre_distribution[genre]} & {round(genre_distribution[genre]/len(y)*100, 2)} & {cumulative} & {round(tokens_per_song[genre], 1)}\\\\")


    ## analyze frequent verses
    # analyze_frequent_verses(x)


def analyze_genre_distribution(y: list):
    genre_distribution = defaultdict(int)
    for genre in y:
        genre_distribution[genre] += 1

    print("\nGenre Distribution: ")
    print_dict(genre_distribution, sorted=True)
    return genre_distribution

def analyze_token_numbers(x: list, y: list, genre_distribution: dict):
    # analyse token and verse numbers per genre
    tokens_per_song = defaultdict(int)
    verses_per_song = defaultdict(int)
    
    all_tokens = 0
    for lyrics, genre in zip(x, y):
        all_tokens += len(lyrics.split(" "))
        tokens_per_song[genre] += len(lyrics.split(" "))
        verses_per_song[genre] += len(lyrics.split("\n"))

    print("\navg. # tokens per song (all Genres):")
    print(round(all_tokens/len(x), 1))

    for key in tokens_per_song.keys():
        tokens_per_song[key] = tokens_per_song[key]/genre_distribution[key]
    for key in verses_per_song.keys():
        verses_per_song[key] = verses_per_song[key]/genre_distribution[key]

    print("\navg. # tokens per song:")
    print_dict(tokens_per_song, sorted=True)
    print("\navg. # verses per song:")
    print_dict(verses_per_song, sorted=True)
    return tokens_per_song


# analyze frequent verses to know which verses to filter out ("Chorus", ...)
def analyze_frequent_verses(x: list):
    verses_dict = defaultdict(int)
    print("\nFrequent verses after initial processing: ")
    for lyrics in x:
        verses = lyrics_to_verses(lyrics)
        for verse in verses:
            verses_dict[verse] += 1
    sort_dict(verses_dict)
    verses_dict = dict(list(verses_dict.items())[:20])
    print_dict(verses_dict, sorted=True)

if __name__ == '__main__':
    main()
