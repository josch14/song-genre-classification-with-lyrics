"""
Glove embedding from Kaggle:
https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation/code
"""
import os
from tqdm import tqdm
import more_itertools

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# local imports
from lib.dataset import Dataset
from lib.utils import read_glove_vector, get_glove_embedding, get_glove_dim, get_early_stopping_callback, get_class_weights, preprocessing
from constants import DATA_FOLDER


class MLP_Glove:
    def __init__(self, 
                ds: Dataset, 
                glove_filename: str,
                learning_rate: float, 
                batch_size: int = 32,
                vocab_size: int = 10000, 
                max_length: int = 256,
                dropout: float = 0.2,
                epochs: int = 1000, 
                early_stop_epochs: int = 10):
    
        # preprocessing pipeline 
        preprocessing_pipeline = [
            "remove_artists", # required; cares about case -> (if used) lower case afterwards
            "lower_case",
            "remove_symbols",
            "remove_contractions",
            "remove_stopwords",
            "remove_whitespaces"]
            
        # dataset
        self.x_train, self.y_train, self.x_test, self.y_test = ds.to_numpy_dataset() 
        self.x_train = preprocessing(self.x_train, preprocessing_pipeline)
        self.x_test = preprocessing(self.x_test, preprocessing_pipeline)
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.x_train)
        self.x_train = tokenizer.texts_to_sequences(self.x_train)
        self.x_train = pad_sequences(self.x_train, maxlen=max_length, padding='post')
        self.x_test = tokenizer.texts_to_sequences(self.x_test)
        self.x_test = pad_sequences(self.x_test, maxlen=max_length, padding='post')

        # training paramaters
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.class_weights = get_class_weights(self.y_train)

        # build glove embedding with embedding matrix
        words_to_index = tokenizer.word_index
        word_to_vec_map = read_glove_vector(os.path.join(DATA_FOLDER, glove_filename))
        glove_embedding = get_glove_embedding(get_glove_dim(glove_filename), max_length, word_to_vec_map, words_to_index)

        # model definition
        self.model = tf.keras.Sequential([
            glove_embedding,
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(ds.n_target_genres, activation='softmax')
        ])

        # model compile
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['accuracy'])

    def train(self):
        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            validation_data=(self.x_test, self.y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[get_early_stopping_callback(self.early_stop_epochs)],
            class_weight=self.class_weights,
            verbose=2)

        print("Calculating predictions on the test set ..")
        batches = more_itertools.chunked(self.x_test, 128)
        test_set_predictions = []
        for batch in tqdm(batches):
            model_out = self.model(tf.constant(batch))
            predictions = [int(tf.math.argmax(out).numpy()) for out in model_out]
            test_set_predictions += predictions
        return test_set_predictions, history