"""
Glove embedding from Kaggle:
https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation/code
"""
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Bidirectional
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
import tensorflow as tf
from lib.dataset import Dataset
from lib.utils import read_glove_vector, get_glove_embedding, get_glove_dim, get_early_stopping_callback
from constants import DATA_FOLDER
from sklearn.utils.class_weight import compute_class_weight
import more_itertools
from tqdm import tqdm
import numpy as np

class LSTM_Glove:
    def __init__(self, ds: Dataset, batch_size: int,  # dataset paramaters
                 vocab_size: int, max_length: int, lstm_dim: int, dropout: float,  # model paramaters
                 epochs: int, learning_rate: float, early_stop_epochs: int,  # training paramaters
                 glove_filename: str):

        # training paramaters
        self.x_train, self.y_train, self.x_test, self.y_test = ds.to_numpy_dataset()
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.class_weights = get_class_weights(ds)


        # Tokenizer stuff
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.x_train)
        words_to_index = tokenizer.word_index
        self.x_train_indices = tokenizer.texts_to_sequences(self.x_train)
        self.x_train_indices = pad_sequences(self.x_train_indices, maxlen=max_length, padding='post')
        self.x_test_indices = tokenizer.texts_to_sequences(self.x_test)
        self.x_test_indices = pad_sequences(self.x_test_indices, maxlen=max_length, padding='post')


        # emebdding matrix
        word_to_vec_map = read_glove_vector(os.path.join(DATA_FOLDER, glove_filename))
        glove_embedding = get_glove_embedding(get_glove_dim(glove_filename), max_length, word_to_vec_map, words_to_index)

        # model
        x_indices = Input(shape=(max_length, ))
        embeddings = glove_embedding(x_indices)
        X = Bidirectional(LSTM(lstm_dim))(embeddings)
        X = Dropout(dropout)(X)
        X = Dense(lstm_dim, activation='relu')(X)
        X = Dropout(dropout)(X)
        X = Dense(ds.n_target_genres, activation='softmax')(X)
        self.model = Model(inputs=x_indices, outputs=X)

        # compile
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['accuracy'])

    def train(self):
        history = self.model.fit(
            self.x_train_indices,
            self.y_train,
            validation_data=(self.x_test_indices, self.y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[get_early_stopping_callback(self.early_stop_epochs)],
            class_weight=self.class_weights,
            verbose=2)

        print("Calculating predictions on the test set ..")
        batches = more_itertools.chunked(self.x_test_indices, 128)
        test_set_predictions = []
        for batch in tqdm(batches):
            model_out = self.model(tf.constant(batch))
            predictions = [int(tf.math.argmax(out).numpy())
                           for out in model_out]
            test_set_predictions += predictions
        return test_set_predictions, history


def get_class_weights(ds: Dataset):
    _, train_genres, _, _ = ds.to_numpy_dataset()

    computed_class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.asarray(list(range(0, ds.n_target_genres))),
        y=train_genres)

    class_weights = {}
    for i in range(0, ds.n_target_genres):
        class_weights[i] = computed_class_weights[i]

    return class_weights