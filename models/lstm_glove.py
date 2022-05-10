"""
Glove embedding from Kaggle:
https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation/code
"""
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Bidirectional, Embedding
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
import tensorflow as tf
from lib.dataset import Dataset
from lib.utils import read_glove_vector, get_glove_embedding
from constants import DATA_FOLDER
import sys
sys.path.append("..")


class LSTM_Glove:
    def __init__(self, ds: Dataset, batch_size: int,  # dataset paramaters
                 vocab_size: int, max_length: int, lstm_dim: int, dropout: float,  # model paramaters
                 epochs: int, learning_rate: float, early_stop_epochs: int,  # training paramaters
                 glove_filename: str):

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

        # training paramaters
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.x_train, self.y_train, self.x_test, self.y_test = ds.to_numpy_dataset()

        # Tokenizer stuff
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.x_train)
        words_to_index = tokenizer.word_index
        self.x_train_indices = tokenizer.texts_to_sequences(self.x_train)
        self.x_train_indices = pad_sequences(
            self.x_train_indices, maxlen=max_length, padding='post')
        self.x_test_indices = tokenizer.texts_to_sequences(self.x_test)
        self.x_test_indices = pad_sequences(
            self.x_test_indices, maxlen=max_length, padding='post')

        # emebdding matrix & set up the model
        word_to_vec_map = read_glove_vector(
            os.path.join(DATA_FOLDER, glove_filename))
        glove_embedding = get_glove_embedding(
            glove_dim, max_length, word_to_vec_map, words_to_index)
        self.model = get_lstm_glove_model(
            max_length=max_length,
            glove_embedding=glove_embedding,
            n_target_genres=ds.n_target_genres,
            lstm_dim=lstm_dim,
            dropout=dropout)

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['accuracy'])

    def train(self):
        # train the model with early stopping after 5 epochs of no improvement
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.early_stop_epochs,
            mode='auto',
            verbose=1,
            # restore_best_weights=True, # maybe use later on
        )

        history = self.model.fit(
            self.x_train_indices,
            self.y_train,
            validation_data=(self.x_test_indices, self.y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[callback])

        return history


def get_lstm_glove_model(max_length, glove_embedding: Embedding, n_target_genres: int, lstm_dim: int, dropout: float):
    x_indices = Input(shape=(max_length, ))
    embeddings = glove_embedding(x_indices)
    X = Bidirectional(LSTM(lstm_dim))(embeddings)
    X = Dropout(dropout)(X)
    X = Dense(lstm_dim)(X)
    X = Dropout(dropout)(X)
    X = Dense(n_target_genres, activation='softmax')(X)
    model = Model(inputs=x_indices, outputs=X)
    return model
