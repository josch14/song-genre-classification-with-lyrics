
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# local imports
from lib.dataset import Dataset
from lib.utils import preprocessing

class SVM:
    def __init__(self, ds: Dataset):
        # preprocessing pipeline 
        preprocessing_pipeline = [
            "remove_artists", # required; cares about case -> (if used) lower case afterwards
            "lower_case",
            "remove_symbols",
            "remove_contractions",
            # "remove_stopwords", # not removing stopwords performs better (maybe because of tfidf)
            "remove_whitespaces"]

        # dataset
        self.x_train, self.y_train, self.x_test, _ = ds.to_numpy_dataset() 
        self.x_train = preprocessing(self.x_train, preprocessing_pipeline)
        self.x_test = preprocessing(self.x_test, preprocessing_pipeline)

        self.model = make_pipeline(
            TfidfVectorizer(),
            SVC(
                gamma='auto', 
                kernel='linear' # worse performing options: rbf, sigmoid
            )
        )

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)
        return predictions
