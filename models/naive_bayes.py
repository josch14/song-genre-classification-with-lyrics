from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline

# local imports
from lib.dataset import Dataset
from lib.utils import preprocessing
from constants import NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB, TFIDF_VECTORIZER

class Naive_Bayes:
    def __init__(self, 
            ds: Dataset, 
            classifier_str: str):
        if classifier_str == NAIVE_BAYES_BERNOULLI_NB:
            # classifier
            classifier = BernoulliNB(alpha=0.05, binarize=0.0) # default binarize value

            # vectorizer
            vectorizer = CountVectorizer()

            # preprocessing pipeline 
            preprocessing_pipeline = [
                "remove_artists", # required; cares about case -> (if used) lower case afterwards
                "lower_case",
                "remove_symbols",
                "remove_contractions",
                # "remove_stopwords", # not removing stopwords performs better (maybe because of tfidf)
                "remove_whitespaces"]

        elif classifier_str == NAIVE_BAYES_MULTINOMIAL_NB:
            # classifier
            classifier = MultinomialNB(alpha=0.05)
            
            # vectorizer
            vectorizer = CountVectorizer()

            # preprocessing pipeline 
            preprocessing_pipeline = [
                "remove_artists", # required; cares about case -> (if used) lower case afterwards
                "lower_case",
                "remove_symbols",
                "remove_contractions",
                "remove_stopwords",
                "remove_whitespaces"]
        else:
            exit(f"Classifier {classifier_str} not implemented ..")

        # model definition
        self.model = make_pipeline(
            vectorizer,
            classifier)

        # dataset
        self.x_train, self.y_train, self.x_test, _ = ds.to_numpy_dataset() 
        self.x_train = preprocessing(self.x_train, preprocessing_pipeline)
        self.x_test = preprocessing(self.x_test, preprocessing_pipeline)


    def train(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)
        return predictions
