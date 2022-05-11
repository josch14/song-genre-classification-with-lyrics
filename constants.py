import os
RESULTS_FOLDER = "results"
EVALUATION_FOLDER = "evaluation"
DATA_FOLDER = "data"
# data input
DATA_LYRICS = os.path.join(DATA_FOLDER, "lyrics-data.csv")
DATA_ARTISTS = os.path.join(DATA_FOLDER, "artists-data.csv")
DATA_PROCESSED = os.path.join(DATA_FOLDER, "processed-data.csv")

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

MODELS = ["naive_bayes_bernoulli", "naive_bayes_multinomial", "svm", "mlp_glove", "lstm", "lstm_glove"]
MODEL_2_NAME = {
    "naive_bayes_bernoulli": "Naive Bayes (Bernoulli)",
    "naive_bayes_multinomial": "Naive Bayes (Multinomial)",
    "svm": "SVM",
    "mlp_glove": "Averaged GloVe + Output Layer",
    "lstm": "LSTM",
    "lstm_glove": "LSTM + GloVe"}

NAIVE_BAYES_BERNOULLI_NB = "bernoulli"
NAIVE_BAYES_MULTINOMIAL_NB = "multinomial"
COUNT_VECTORIZER = "count"
TFIDF_VECTORIZER = "tfidf"

SYMBOLS = ["\n", "\r", "!", "”", "\"", "#", "$", "%", "&", "(", ")", \
    "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", \
    "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "’"]


MACRO_PRECISION = "macro-precision"
MACRO_RECALL = "macro-recall"
MACRO_F1 = "macro-f1"
WEIGHTED_PRECISION = "weighted-precision"
WEIGHTED_RECALL = "weighted-recall"
WEIGHTED_F1 = "weighted-f1"

METRIC = [
    MACRO_PRECISION,
    MACRO_RECALL,
    MACRO_F1,
    WEIGHTED_PRECISION,
    WEIGHTED_RECALL,
    WEIGHTED_F1
]

METRIC_2_NAME = {
    MACRO_PRECISION: "Macro-Average Precision",
    MACRO_RECALL: "Macro-Average Recall",
    MACRO_F1: "Macro-Average F1 Score",
    WEIGHTED_PRECISION: "Weighted-Average Precision",
    WEIGHTED_RECALL: "Weighted-Average Recall",
    WEIGHTED_F1: "Weighted-Average F1 Score"}
