from numpy import vectorize
from lib.utils import get_artists, remove_artists, remove_symbols, remove_contractions

from constants import NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB, COUNT_VECTORIZER, TFIDF_VECTORIZER
from lib.dataset import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline


"""
https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
"""
class Naive_Bayes:
    def __init__(self, ds: Dataset, classifier_str: str, vectorizer_str: str):
        if classifier_str == NAIVE_BAYES_BERNOULLI_NB:
            classifier = BernoulliNB()
        elif classifier_str == NAIVE_BAYES_MULTINOMIAL_NB:
            classifier = MultinomialNB()
        else:
            exit(f"Classifier {classifier_str} unknown ..")

        if vectorizer_str == COUNT_VECTORIZER:
            vectorizer = CountVectorizer()
        elif vectorizer_str == TFIDF_VECTORIZER:
            vectorizer = TfidfVectorizer()
        else:
            exit(f"Vectorizer {classifier_str} unknown ..")

        self.model = make_pipeline(
            vectorizer,
            classifier)
        self.ds = ds

        
        # convert text to lowercase
        self.ds.x_train = [lyrics.lower() for lyrics in self.ds.x_train]
        self.ds.x_test = [lyrics.lower() for lyrics in self.ds.x_test]

        # remove artists from lyrics
        artists = get_artists()
        self.ds.x_train = [remove_artists(lyrics, artists) for lyrics in self.ds.x_train]
        self.ds.x_test = [remove_artists(lyrics, artists) for lyrics in self.ds.x_test]

        # remove new lines/carriage returns and symbols
        self.ds.x_train = [remove_symbols(lyrics) for lyrics in self.ds.x_train]
        self.ds.x_test = [remove_symbols(lyrics) for lyrics in self.ds.x_test]

        # remove new lines
        self.ds.x_train = [lyrics.rstrip("\n") for lyrics in self.ds.x_train]
        self.ds.x_test = [lyrics.rstrip("\n") for lyrics in self.ds.x_test]

        # resolve contractions
        self.ds.x_train = [remove_contractions(lyrics) for lyrics in self.ds.x_train]
        self.ds.x_test = [remove_contractions(lyrics) for lyrics in self.ds.x_test]

        # # remove stopwords
        # self.ds.x_train = [remove_stopwords(lyrics) for lyrics in self.ds.x_train]
        # self.ds.x_test = [remove_stopwords(lyrics) for lyrics in self.ds.x_test]


    def train(self):
        self.model.fit(self.ds.x_train, self.ds.y_train)
        predictions = self.model.predict(self.ds.x_test)
        return predictions
