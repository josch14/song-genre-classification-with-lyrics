import sys
sys.path.append("..")

import numpy as np
from constants import DATA_FOLDER
import os
from lib.utils import read_glove_vector
from constants import NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB
from lib.dataset import Dataset


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

"""
https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
"""


class Naive_Bayes_Glove:
    def __init__(self, ds: Dataset, classifier_str: str, glove_filename: str):

        if classifier_str == NAIVE_BAYES_BERNOULLI_NB:
            classifier = BernoulliNB()
        elif classifier_str == NAIVE_BAYES_MULTINOMIAL_NB:
            classifier = MultinomialNB()
        else:
            exit(f"Naive Bayes Classifier {classifier_str} unknown ..")
        classifier = SVC()

        if "50d" in glove_filename:
            glove_dim = 50
        elif "100d" in glove_filename:
            glove_dim = 100
        elif "200d" in glove_filename:
            glove_dim = 200
        elif "300d" in glove_filename:
            glove_dim = 300
        else:
            exit("Glove Dimension not valid. Exiting ...")

        self.ds = ds


        # emebdding matrix & set up the model
        word_to_vec_map = read_glove_vector(
            os.path.join(DATA_FOLDER, glove_filename))

        self.model = make_pipeline(
            W2vVectorizer(word_to_vec_map, glove_dim),
            classifier)

    def train(self):
        self.model.fit(self.ds.x_train, self.ds.y_train)
        predictions = self.model.predict(self.ds.x_test)
        return predictions


class W2vVectorizer(object):

    def __init__(self, w2v, glove_dim):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        self.glove_dim = glove_dim

    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline
    def fit(self, X, y):
        return self

    def transform(self, X):
        # print(X)
        from tqdm import tqdm
        means = []
        for lyrics in tqdm(X):
            verses = lyrics.split("\n")
            text = "\n".join(verses)
            text = text.replace("   ", "  ")
            text = text.replace("  ", " ")
            words = text.split(" ")
            words = [self.w2v[w] for w in words if w in self.w2v]
            words = np.array(words)
            if len(words) == 0:
                words = np.zeros(shape=(1, 50))
            text_mean = words.mean(axis=0)
            means.append(text_mean)
        return means