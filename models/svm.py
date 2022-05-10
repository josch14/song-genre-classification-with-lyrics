from lib.dataset import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
"""
https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
"""


class SVM:
    def __init__(self, ds: Dataset):
        self.model = make_pipeline(
            TfidfVectorizer(),
            SVC(gamma='auto', kernel='linear'))  # worse performing options: rbf, sigmoid
        self.ds = ds

    def train(self):
        self.model.fit(self.ds.x_train, self.ds.y_train)
        predictions = self.model.predict(self.ds.x_test)
        return predictions
