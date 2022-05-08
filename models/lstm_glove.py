"""
Glove embedding from Kaggle:
https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation/code
"""
import sys
sys.path.append("..")
import os 
from constants import DATA_FOLDER, GLOVE_FILENAME

from lib.dataset import Dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input,Bidirectional, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



class LSTM_Glove:
    def __init__(self, ds: Dataset, batch_size: int, # dataset paramaters
        vocab_size: int, max_length: int, lstm_dim: int, dropout: float, # model paramaters
        epochs: int, learning_rate: float, early_stop_epochs: int # training paramaters
        ):
        
        # training paramaters
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.x_train, self.y_train, self.x_val, self.y_val = ds.to_numpy_dataset()

        # Tokenizer stuff
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(self.x_train)
        words_to_index = tokenizer.word_index
        self.x_train_indices = tokenizer.texts_to_sequences(self.x_train)
        self.x_train_indices = pad_sequences(self.x_train_indices, maxlen=max_length, padding='post')
        self.x_val_indices = tokenizer.texts_to_sequences(self.x_val)
        self.x_val_indices = pad_sequences(self.x_val_indices, maxlen=max_length, padding='post')

        # emebdding matrix & set up the model
        word_to_vec_map = read_glove_vector(os.path.join(DATA_FOLDER, GLOVE_FILENAME))
        glove_embedding = get_glove_embedding(max_length, word_to_vec_map, words_to_index)
        self.model = get_lstm_glove_model(
            max_length=max_length,
            glove_embedding = glove_embedding,
            n_target_genres=ds.n_target_genres,
            lstm_dim=lstm_dim,
            dropout=dropout)

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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
            validation_data=(self.x_val_indices, self.y_val), 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            callbacks=[callback])

        return history

def get_lstm_glove_model(max_length, glove_embedding: Embedding, n_target_genres: int, lstm_dim: int, dropout: float):
    x_indices = Input(shape = (max_length, ))
    embeddings = glove_embedding(x_indices)
    X = Bidirectional(LSTM(lstm_dim))(embeddings)
    X = Dropout(dropout)(X)
    X = Dense(n_target_genres, activation='softmax')(X)
    model = Model(inputs=x_indices, outputs=X)
    return model

"""
Use Glove.
https://towardsdatascience.com/sentiment-analysis-using-lstm-and-glove-embeddings-99223a87fe8e
"""
def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
        return word_to_vec_map


def get_glove_embedding(max_length: int, word_to_vec_map: dict, words_to_index: dict):
    vocab_len = len(words_to_index)
    print(vocab_len)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index-1, :] = embedding_vector

    return Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=max_length, weights = [emb_matrix], trainable=False)