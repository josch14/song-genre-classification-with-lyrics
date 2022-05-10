
import sys
sys.path.append("..")

from lib.dataset import Dataset
import tensorflow as tf
import more_itertools
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class LSTM:
    def __init__(self, ds: Dataset, batch_size: int,  # dataset paramaters
                 vocab_size: int, max_length: int, lstm_dim: int, dropout: float,  # model paramaters
                 epochs: int, learning_rate: float, early_stop_epochs: int  # training paramaters
                 ):

        # training paramaters
        self.epochs = epochs
        self.early_stop_epochs = early_stop_epochs
        self.ds = ds
        self.train_dataset, self.test_dataset = ds.to_tf_dataset(batch_size)

        # compute class weights since the genres of song occur not equally often
        self.class_weights = get_class_weights(ds)

        # training paramaters
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.x_train, self.y_train, self.x_test, self.y_test = ds.to_numpy_dataset()

        # Tokenizer stuff
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.x_train)

        self.x_train_indices = tokenizer.texts_to_sequences(self.x_train)
        self.x_train_indices = pad_sequences(
            self.x_train_indices, maxlen=max_length, padding='post')
        self.x_test_indices = tokenizer.texts_to_sequences(self.x_test)
        self.x_test_indices = pad_sequences(
            self.x_test_indices, maxlen=max_length, padding='post')

        # model itself
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=lstm_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(lstm_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(ds.n_target_genres, activation='softmax')
        ])
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['accuracy'])

    def train(self):

        # use early stopping
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.early_stop_epochs,
            mode='auto',
            verbose=1,
            restore_best_weights=True,  # maybe use later on
        )

        history = self.model.fit(
            self.x_train_indices,
            self.y_train,
            validation_data=(self.x_test_indices, self.y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[callback],
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