RESULTS_FOLDER = "results"
DATA_FOLDER = "data"

# Define the genres in order they join the dataset
TARGET_GENRES = [
    "Pop", 
    "Rock", 
    "Rap", 
    "Country", 
    "Reggae", 
    "Heavy Metal", 
    "Blues", 
    "Indie",
    "Hip Hop", 
    "Jazz", 
    "Folk", 
    "Gospel/Religioso", 
]

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
    

GLOVE_FILENAME_42B_300D = "glove.42B.300d.txt"
GLOVE_FILENAME_6B_50D = "glove.6B.50d.txt"
GLOVE_FILENAME_6B_100D = "glove.6B.100d.txt"
GLOVE_FILENAME_6B_200D = "glove.6B.200d.txt"
GLOVE_FILENAME_6B_300D = "glove.6B.300d.txt"

MODELS = ["lstm", "lstm_glove", "naive_bayes_bernoulli", "naive_bayes_multinomial", "svm"]

NAIVE_BAYES_BERNOULLI_NB = "bernoulli"
NAIVE_BAYES_MULTINOMIAL_NB = "multinomial"
