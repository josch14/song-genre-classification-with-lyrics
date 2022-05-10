from lib.dataset import Dataset
from collections import defaultdict
from lib.utils import print_dict


def main():
    dataset = Dataset(n_target_genres=12)

    x = dataset.x_train + dataset.x_test
    y = dataset.y_train + dataset.y_test

    # analyze genre distribution
    genre_distribution = analyze_genre_distribution(y)

    # songs_per_genre = len(x)/12

    # # analyse token and verse numbers per genre
    # tokens_per_song = defaultdict(int)
    # verses_per_song = defaultdict(int)

    # for lyrics, genre in zip(x, y):
    #     tokens_per_song[genre] += len(lyrics.split(" "))
    #     verses_per_song[genre] += len(lyrics.split("\n"))

    # for key in tokens_per_song.keys():
    #     tokens_per_song[key] = tokens_per_song[key]/songs_per_genre
    # for key in verses_per_song.keys():
    #     verses_per_song[key] = verses_per_song[key]/songs_per_genre

    # print("\navg. # tokens per song:")
    # print_dict(tokens_per_song, sorted=True)
    # print("\navg. # verses per song:")
    # print_dict(verses_per_song, sorted=True)

    # # frequent verses (--> Filter out things like CHORUS)
    # verses_dict = defaultdict(int)
    # print("\nFrequent verses after initial processing: ")
    # for lyrics in x:
    #     verses = lyrics_to_verses(lyrics)
    #     for verse in verses:
    #         verses_dict[verse] += 1
    # sort_dict(verses_dict)
    # verses_dict = dict(list(verses_dict.items())[:20])
    # print_dict(verses_dict, sorted=True)


def analyze_genre_distribution(y: list):
    genre_distribution = defaultdict(int)
    for genre in y:
        genre_distribution[genre] += 1

    print("\nGenre Distribution: ")
    print_dict(genre_distribution, sorted=True)
    return genre_distribution


if __name__ == '__main__':
    main()
