from lib.dataset import Dataset
import tensorflow as tf

class LSTM:
    def __init__(self, ds: Dataset, batch_size: int, # dataset paramaters
        vocab_size: int, embedding_dim: int, dropout: float, # model paramaters
        epochs: int, learning_rate: float, early_stop_epochs: int # training paramaters
        ):
        
        # training paramaters
        self.epochs = epochs
        self.early_stop_epochs = early_stop_epochs
        self.train_dataset, self.test_dataset = ds.to_tf_dataset(batch_size)

        # model itself
        self.model = tf.keras.Sequential([
            get_text_encoder(self.train_dataset, vocab_size),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True), # use masking to handle the variable sequence lengths
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(ds.n_target_genres, activation='softmax')
        ])
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
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.test_dataset,
            callbacks=[callback])


        return history


"""
Transforms strings into vocabulary indices.
"""
def get_text_encoder(train_split, vocab_size):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_split.map(lambda text, label: text))
    return encoder

