from lib.dataset import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import sys
sys.path.append("..")
from constants import NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB

"""
https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
"""
class Naive_Bayes:
    def __init__(self, ds: Dataset, classifier_str: str):
        if classifier_str == NAIVE_BAYES_BERNOULLI_NB:
            classifier = BernoulliNB()
        elif classifier_str == NAIVE_BAYES_MULTINOMIAL_NB:
            classifier = MultinomialNB()
        else:
            exit(f"Naive Bayes Classifier {classifier_str} unknown ..")

        self.model = make_pipeline(
            TfidfVectorizer(), 
            classifier)
        self.ds = ds

    def train(self):
        self.model.fit(self.ds.x_train, self.ds.y_train)
        predictions = self.model.predict(self.ds.x_val)

        val_accuracy = accuracy_score(self.ds.y_val, predictions)
        print(f"The accuracy is {val_accuracy}")
        return val_accuracy