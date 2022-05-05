from lib.dataset import Dataset

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class LSTM:
    def __init__(self, ds: Dataset, batch_size:int=32, vocab_size:int=10000, embedding_dim:int=64, dropout:float=0.3):
        
        self.train_dataset, self.val_dataset = ds.to_tf_dataset(batch_size)
        self.model = get_model(
            n_target_genres=ds.n_target_genres,
            train_dataset=self.train_dataset,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy'])

    def train(self):
        history = self.model.fit(
            self.train_dataset,
            epochs=10,
            validation_data=self.val_dataset)
        return history

def get_model(n_target_genres: int, train_dataset: tf.data.Dataset, vocab_size:int=10000, embedding_dim:int=64, dropout:float=0.3):
    return tf.keras.Sequential([
        get_text_encoder(train_dataset, vocab_size),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(n_target_genres, activation='softmax')
    ])

#  transforms strings into vocabulary indices
def get_text_encoder(train_split, vocab_size):
    encoder = TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_split.map(lambda text, label: text))
    return encoder

