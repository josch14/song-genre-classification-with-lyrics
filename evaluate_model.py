import numpy as np
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn

from models import *
from lib.dataset import *
from lib.utils import *
from constants import *

import matplotlib.pyplot as plt
from matplotlib import figure
# train LSTM
def train_lstm(dataset: Dataset, learning_rate: float):
    model = LSTM(
        ds=dataset,
        learning_rate=learning_rate)
    test_set_predictions, history = model.train()
    return test_set_predictions, history

# train LSTM with GloVe
def train_lstm_glove(dataset: Dataset, learning_rate: float):
    model = LSTM_Glove(
        ds=dataset,
        learning_rate=learning_rate,
        glove_filename=GLOVE_FILENAME_6B_100D)
    test_set_predictions, history = model.train()
    return test_set_predictions, history

# train MLP with GloVe
def train_mlp_glove(dataset: Dataset, learning_rate: float):
    model = MLP_Glove(
        ds=dataset,
        learning_rate=learning_rate,
        glove_filename=GLOVE_FILENAME_6B_50D)
    test_set_predictions, history = model.train()
    return test_set_predictions, history

# train Naive Bayes
def naive_bayes(dataset: Dataset, classifier_str: str):
    model = Naive_Bayes(
        ds=dataset, 
        classifier_str=classifier_str)
    test_set_predictions = model.train()
    return test_set_predictions

# train SVM
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
parser.add_argument('-lr', '--learning_rate', default=None, type=float, help='Learning rate for lstm and mlp.')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model

    if not model_name in MODELS:
        exit(f"\nError: Model {model_name} is not implemented ..")
    print(f"\nEvaluating Method {model_name} on full dataset (12 genres)..")

    # create results folder if it does not exist
    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)

    n_target_genres = 12 # use all genres
    dataset = Dataset(n_target_genres)
    test_set_predictions = None


    # evaluate desired method
    if model_name == "naive_bayes_bernoulli":
        test_set_predictions = naive_bayes(dataset, NAIVE_BAYES_BERNOULLI_NB)

    elif model_name == "naive_bayes_multinomial":
        test_set_predictions = naive_bayes(dataset, NAIVE_BAYES_MULTINOMIAL_NB)

    elif model_name == "svm":
        test_set_predictions = svm(dataset)

    else:
        if args.learning_rate is None:
            exit(f"\nError: For this model, learning rate needs to be specified ..")
        learning_rate = args.learning_rate

        if model_name == "mlp_glove":
            test_set_predictions, _ = train_mlp_glove(dataset, learning_rate)

        elif model_name == "lstm":
            test_set_predictions, _ = train_lstm(dataset, learning_rate)

        elif model_name == "lstm_glove":
            test_set_predictions, _ = train_lstm_glove(dataset, learning_rate)

    test_set_predictions = [LABEL_2_GENRE[p] for p in test_set_predictions]
    report = classification_report(dataset.y_test, test_set_predictions, digits=4)
    report_dict = classification_report(dataset.y_test, test_set_predictions, digits=4, output_dict=True)

    # print macro-averages
    stats = report_dict["macro avg"]
    print(f"[Macro]    P: {round_float(stats['precision']*100)}   "
            + f"R: {round_float(stats['recall']*100)}   "
            + f"F1: {round_float(stats['f1-score']*100)}")

    # print weighted-averages
    stats = report_dict["weighted avg"]
    print(f"[Weighted] P: {round_float(stats['precision']*100)}   "
            + f"R: {round_float(stats['recall']*100)}   "
            + f"F1: {round_float(stats['f1-score']*100)}")

    cm = confusion_matrix(dataset.y_test, test_set_predictions)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2) 
    df_cm = pd.DataFrame(cm, index = [i for i in TARGET_GENRES[:n_target_genres]], columns = [i for i in TARGET_GENRES[:n_target_genres]])

    figure.Figure(figsize = (19.2, 15))
    sn.set(font_scale=0.55) # Adjust to fit
    svm = sn.heatmap(df_cm, annot=True,cmap="Blues")
    plt.title(MODEL_2_NAME[model_name], fontweight='bold')
    plt.xlabel("\nPredicted Category", fontweight='bold')
    plt.ylabel("Target Category", fontweight='bold')
    figure = svm.get_figure()


    # save classification report and confusion matrix to local files
    text_file = open(
        os.path.join(EVALUATION_FOLDER, 
        f"{model_name}.txt"), 
        "w")
    text_file.write(report)
    text_file.close()

    figure.savefig(
        os.path.join(EVALUATION_FOLDER, f"{model_name}.png"), 
        dpi=300, bbox_inches = 'tight', 
        pad_inches=0.0)
