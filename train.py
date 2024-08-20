import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# Funkcja do wczytywania danych z obsługą błędów
def load_data(file_path, delimiter='\t', header=None):
    try:
        return pd.read_csv(file_path, delimiter=delimiter, header=header, quoting=3)
    except pd.errors.ParserError as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()  # Zwróć pustą ramkę danych w przypadku błędu

# Wczytanie danych
train_data = load_data('train/in_1.csv', delimiter=',')
train_labels = load_data('train/expected.tsv')
dev_data = load_data('dev-0/in_1.csv', delimiter=',')
dev_labels = load_data('dev-0/expected.tsv')
test_data = load_data('test-A/in_1.csv', delimiter=',')

# Sprawdzenie, czy dane zostały poprawnie wczytane
if train_data.empty or train_labels.empty or dev_data.empty or dev_labels.empty or test_data.empty:
    raise ValueError("One or more data files could not be loaded correctly. Please check the data files.")

# Przygotowanie danych
def prepare_data(data):
    texts = data[0] + ' [MASK] ' + data[1]
    return texts

train_texts = prepare_data(train_data)
dev_texts = prepare_data(dev_data)
test_texts = prepare_data(test_data)

# Tokenizacja i sekwencjonowanie
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

def text_to_sequences(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

max_len = max([len(seq) for seq in tokenizer.texts_to_sequences(train_texts)])

X_train = text_to_sequences(train_texts, tokenizer, max_len)
X_dev = text_to_sequences(dev_texts, tokenizer, max_len)
X_test = text_to_sequences(test_texts, tokenizer, max_len)

# Przygotowanie danych wyjściowych (y_train i y_dev)
# Przygotowanie danych wyjściowych (y_train i y_dev)
def words_to_sequences(words, tokenizer):
    sequences = []
    for word in words:
        if not isinstance(word, str):  # Jeśli wartość nie jest typu string
            word = str(word)  # Konwersja na string
        sequences.append(tokenizer.texts_to_sequences([word])[0])
    return sequences

# Przygotowanie danych wyjściowych (y_train i y_dev)
y_train_seq = words_to_sequences(train_labels[0], tokenizer)
y_dev_seq = words_to_sequences(dev_labels[0], tokenizer)

# Pad sekwencje wyjściowe
max_label_len = max(len(seq) for seq in y_train_seq + y_dev_seq)
y_train_seq = pad_sequences(y_train_seq, maxlen=max_label_len, padding='post')
y_dev_seq = pad_sequences(y_dev_seq, maxlen=max_label_len, padding='post')

# Konwersja y_train_seq i y_dev_seq na one-hot encodings
y_train_seq = to_categorical(y_train_seq, num_classes=len(tokenizer.word_index) + 1)
y_dev_seq = to_categorical(y_dev_seq, num_classes=len(tokenizer.word_index) + 1)


vocab_size = len(tokenizer.word_index) + 1  # Rozmiar słownika

# Definicja modelu GRU
embedding_dim = 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(GRU(64)),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Custom callback to display training progress
class TrainingProgress(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()  # Zapisujemy czas rozpoczęcia trenowania
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time  # Obliczamy czas trwania epoki
        remaining_time = epoch_time * (self.params['epochs'] - epoch - 1)  # Obliczamy szacowany czas pozostały do zakończenia trenowania
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f} - Time elapsed: {epoch_time:.2f}s - Remaining time: {remaining_time:.2f}s")


# Trenowanie modelu
history = model.fit(
    X_train, y_train_seq,
    epochs=10,
    batch_size=64,
    validation_data=(X_dev, y_dev_seq),
    callbacks=[TrainingProgress()]
)

# Przewidywanie i zapis wyników dla dev-0
dev_predictions = model.predict(X_dev)
dev_predicted_words = [tokenizer.index_word.get(idx, '') for idx in np.argmax(dev_predictions, axis=1)]

with open('dev-0/out.tsv', 'w') as out_file:
    for word in dev_predicted_words:
        out_file.write(f"{word}\n")

# Przewidywanie i zapis wyników dla test-A
test_predictions = model.predict(X_test)
test_predicted_words = [tokenizer.index_word.get(idx, '') for idx in np.argmax(test_predictions, axis=1)]

with open('test-A/out.tsv', 'w') as out_file:
    for word in test_predicted_words:
        out_file.write(f"{word}\n")
