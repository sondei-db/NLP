import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.callbacks import Callback
import time

# Function to load data from CSV files
def load_data(file_path, delimiter=','):
    return pd.read_csv(file_path, delimiter=delimiter, encoding= 'utf-8', header = None, on_bad_lines='warn')

# Custom callback to display remaining time
class RemainingTimeCallback(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        remaining_epochs = self.params['epochs'] - epoch - 1
        remaining_time = remaining_epochs * epoch_time
        print(f'Epoch {epoch+1}/{self.params["epochs"]} - Remaining time: {remaining_time:.2f} seconds')

# Load data
train_data = load_data('train/in_1.csv', delimiter=',')
train_labels = load_data('train/expected.tsv', delimiter='\t')
#dev_data = load_data('dev-0/in_1.csv', delimiter=',')
#dev_labels = load_data('dev-0/expected.tsv', delimiter='\t')
test_data = load_data('test-A/in_1.csv', delimiter=',')

# Print information about labels data
print("Training labels info:")
print(train_labels.info())
print("\nDevelopment labels info:")
#print(dev_labels.info())

# Print a sample of labels data
print("\nSample of training labels:")
print(train_labels.head())
print("\nSample of development labels:")
#print(dev_labels.head())

# Preprocess data
tokenizer = Tokenizer()
train_data_text = train_data[0] + ' ' + train_data[1]
tokenizer.fit_on_texts(train_data_text)
vocab_size = len(tokenizer.word_index) + 1
max_sequence_length = max(train_data_text.apply(len))

# Convert text data to sequences
train_data_seq = tokenizer.texts_to_sequences(train_data_text)
train_data_seq_padded = pad_sequences(train_data_seq, maxlen=max_sequence_length)
train_labels_seq = tokenizer.texts_to_sequences(train_labels['A'])
train_labels_seq_padded = pad_sequences(train_labels_seq, maxlen=1)


# Define model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length))
model.add(GRU(units=128, return_sequences=True))
model.add(GRU(units=64))
model.add(Dense(units=vocab_size, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callback for remaining time
remaining_time_callback = RemainingTimeCallback()

# Train model
model.fit([train_data_seq1_padded, train_data_seq2_padded], train_labels_seq_padded,
          epochs=10, batch_size=32,
          validation_data=([dev_data_seq1_padded, dev_data_seq2_padded], dev_labels_seq_padded),
          callbacks=[remaining_time_callback])

# Generate predictions for development set
dev_predictions = model.predict([dev_data_seq1_padded, dev_data_seq2_padded])
dev_predicted_words = [tokenizer.index_word[np.argmax(prediction)] for prediction in dev_predictions]

# Write predictions to file for development set
dev_output = pd.DataFrame({'predicted_word': dev_predicted_words})
dev_output.to_csv('dev_predictions.tsv', sep='\t', index=False)

# Generate predictions for test set
test_predictions = model.predict([test_data_seq1_padded, test_data_seq2_padded])
test_predicted_words = [tokenizer.index_word[np.argmax(prediction)] for prediction in test_predictions]

# Write predictions to file for test set
test_output = pd.DataFrame({'predicted_word': test_predicted_words})
test_output.to_csv('test_predictions.tsv', sep='\t', index=False)
