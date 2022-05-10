from cgi import test
from models.lstm import LSTM
from models.lstm_glove import LSTM_Glove
from models.naive_bayes import Naive_Bayes
from models.svm import SVM
from models.mlp_glove import MLP_Glove
# from models.naive_bayes_glove import Naive_Bayes_Glove
from lib.dataset import Dataset
import numpy as np
import argparse
import os
from constants import LABEL_2_GENRE, MODELS, RESULTS_FOLDER, GLOVE_FILENAME_42B_300D, \
    GLOVE_FILENAME_6B_50D, GLOVE_FILENAME_6B_100D, GLOVE_FILENAME_6B_200D, GLOVE_FILENAME_6B_300D, NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from lib.utils import print_dict, round_float


def train_lstm(dataset: Dataset, learning_rate: float):
    model = LSTM(
        ds=dataset,
        batch_size=32,
        vocab_size=10000,
        max_length=256,
        lstm_dim=64,
        dropout=0.2,
        learning_rate=learning_rate,
        epochs=1000,
        early_stop_epochs=10)
    test_set_predictions, history = model.train()
    return test_set_predictions, history


def train_lstm_glove(dataset: Dataset, learning_rate: float):
    model = LSTM_Glove(
        ds=dataset,
        batch_size=32,
        vocab_size=10000,
        max_length=256,
        lstm_dim=64,
        dropout=0.2,
        learning_rate=learning_rate,
        epochs=1000,
        early_stop_epochs=10,
        glove_filename=GLOVE_FILENAME_6B_100D)
    test_set_predictions, history = model.train()
    return test_set_predictions, history


def train_mlp_glove(dataset: Dataset, learning_rate: float):
    model = MLP_Glove(
        ds=dataset,
        batch_size=32,
        vocab_size=10000,
        max_length=256,
        dropout=0.2,
        learning_rate=learning_rate,
        epochs=1000,
        early_stop_epochs=10,
        glove_filename=GLOVE_FILENAME_6B_50D)
    test_set_predictions, history = model.train()
    return test_set_predictions, history


# def naive_bayes_glove(dataset: Dataset, classifier_str: str):
#     model = Naive_Bayes_Glove(dataset, classifier_str,
#                               glove_filename=GLOVE_FILENAME_6B_100D)
#     test_set_predictions = model.train()
#     return test_set_predictions


def naive_bayes(dataset: Dataset, classifier_str: str):
    model = Naive_Bayes(dataset, classifier_str)
    test_set_predictions = model.train()
    return test_set_predictions


def svm(dataset: Dataset):
    model = SVM(dataset)
    test_set_predictions = model.train()
    return test_set_predictions


"""
Example call: 
python train.py -m lstm
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help='Model to train.')
parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate for lstm and mlp.')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model

    # create results folder if it does not exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # train for different numbers of target genres
    for n_target_genres in range(2, 12+1):
        if not model_name in MODELS:
            print(f"\nModel {model_name} is not implemented ..")
            exit()
        print(f"\nMethod: {model_name}, # genres: {n_target_genres}")

        dataset = Dataset(n_target_genres)
        save_path = os.path.join(
            RESULTS_FOLDER, f"{model_name}_{n_target_genres}.npy")
        test_set_predictions = None

        # call desired method
        if model_name == "naive_bayes_bernoulli":
            test_set_predictions = naive_bayes(
                dataset, NAIVE_BAYES_BERNOULLI_NB)

        elif model_name == "naive_bayes_multinomial":
            test_set_predictions = naive_bayes(
                dataset, NAIVE_BAYES_MULTINOMIAL_NB)

        elif model_name == "svm":
            test_set_predictions = svm(dataset)

        else:
            learning_rate = args.learning_rate
            if model_name == "mlp_glove":
                test_set_predictions, history = train_mlp_glove(dataset, learning_rate)
            elif model_name == "lstm":
                test_set_predictions, history = train_lstm(dataset, learning_rate)
            elif model_name == "lstm_glove":
                test_set_predictions, history = train_lstm_glove(dataset, learning_rate)
            else: 
                continue

            test_set_predictions = [LABEL_2_GENRE[p] for p in test_set_predictions]
            np.save(save_path, history.history)

        report = classification_report(
            dataset.y_test, test_set_predictions, digits=4, output_dict=True)
        # print macro-averages
        stats = report["macro avg"]
        print(f"[Macro]    P: {round_float(stats['precision']*100)}   "
              + f"R: {round_float(stats['recall']*100)}   "
              + f"F1: {round_float(stats['f1-score']*100)}")
        # print weighted-averages
        stats = report["weighted avg"]
        print(f"[Weighted] P: {round_float(stats['precision']*100)}   "
              + f"R: {round_float(stats['recall']*100)}   "
              + f"F1: {round_float(stats['f1-score']*100)}")
