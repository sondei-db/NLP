import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical

# Load Data
def load_data(file_path, delimiter=','):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(line.strip().split(delimiter))
            except Exception as e:
                print(f"Error processing line: {line}. Error: {e}")
    return pd.DataFrame(data)

train_data = load_data('train/in_1.csv', delimiter=',')
train_labels = load_data('train/expected.tsv', delimiter='\t')
dev_data = load_data('dev-0/in_1.csv', delimiter=',')
dev_labels = load_data('dev-0/expected.tsv', delimiter='\t')
test_data = load_data('test-A/in_1.csv', delimiter=',')

# Text Preprocessing
tokenizer = Tokenizer()
all_texts = train_data.iloc[:, 0].tolist() + train_data.iloc[:, 1].tolist()
tokenizer.fit_on_texts(all_texts)

train_sequences1 = tokenizer.texts_to_sequences(train_data.iloc[:, 0])
train_sequences2 = tokenizer.texts_to_sequences(train_data.iloc[:, 1])
dev_sequences1 = tokenizer.texts_to_sequences(dev_data.iloc[:, 0])
dev_sequences2 = tokenizer.texts_to_sequences(dev_data.iloc[:, 1])
test_sequences1 = tokenizer.texts_to_sequences(test_data.iloc[:, 0])
test_sequences2 = tokenizer.texts_to_sequences(test_data.iloc[:, 1])

max_length = max([len(seq) for seq in train_sequences1 + train_sequences2])
train_sequences1 = pad_sequences(train_sequences1, maxlen=max_length, padding='post')
train_sequences2 = pad_sequences(train_sequences2, maxlen=max_length, padding='post')
dev_sequences1 = pad_sequences(dev_sequences1, maxlen=max_length, padding='post')
dev_sequences2 = pad_sequences(dev_sequences2, maxlen=max_length, padding='post')
test_sequences1 = pad_sequences(test_sequences1, maxlen=max_length, padding='post')
test_sequences2 = pad_sequences(test_sequences2, maxlen=max_length, padding='post')

# Prepare Labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(train_labels.iloc[:, 0])

train_labels_seq = label_tokenizer.texts_to_sequences(train_labels.iloc[:, 0])
dev_labels_seq = label_tokenizer.texts_to_sequences(dev_labels.iloc[:, 0])

# Padding Labels
max_label_length = max(len(label) for label in train_labels_seq)
train_labels_seq = pad_sequences(train_labels_seq, maxlen=max_label_length, padding='post')
dev_labels_seq = pad_sequences(dev_labels_seq, maxlen=max_label_length, padding='post')

print("Shape of train_labels_seq:", train_labels_seq.shape)
print("Shape of dev_labels_seq:", dev_labels_seq.shape)

num_classes = max_label_length

# Define Callback for Estimated Time Monitoring
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        print(f"Epoch {epoch+1} time: {self.times[-1]:.2f} seconds")

# Build GRU Model
embedding_dim = 128
gru_units = 64

input1 = Input(shape=(max_length,))
input2 = Input(shape=(max_length,))
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)
embedded1 = embedding(input1)
embedded2 = embedding(input2)
gru = GRU(gru_units)
output1 = gru(embedded1)
output2 = gru(embedded2)
merged = Concatenate()([output1, output2])
dense = Dense(64, activation='relu')(merged)
output = Dense(8, activation='softmax')(dense)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_labels_seq = np.argmax(train_labels_seq, axis=1)
dev_labels_seq = np.argmax(dev_labels_seq, axis=1)
print("Shape of train_labels_seq:", train_labels_seq.shape)
print("Shape of dev_labels_seq:", dev_labels_seq.shape)


# Train the Model
time_callback = TimeHistory()
history = model.fit(
    [train_sequences1, train_sequences2],
    train_labels_seq,
    validation_data=([dev_sequences1, dev_sequences2], dev_labels_seq),
    epochs=1,
    batch_size=32,
    callbacks=[time_callback]
)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# Save Predictions to out.tsv
def save_predictions(sequences1, sequences2, label_tokenizer, file_path):
    predictions = model.predict([sequences1, sequences2])
    print(predictions)  # print raw predictions
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_words = label_tokenizer.sequences_to_texts(predicted_labels.reshape(-1, 1))
    if not predicted_words:
        predicted_words = label_tokenizer.sequences_to_texts(predicted_labels)
    df = pd.DataFrame(predicted_words, columns=["predicted_gap"])
    df.to_csv(file_path, sep='\t', index=False)

save_predictions(dev_sequences1, dev_sequences2, label_tokenizer, 'dev-0/out.tsv')
save_predictions(test_sequences1, test_sequences2, label_tokenizer, 'test-A/out.tsv')

# Optionally, evaluate on dev set
dev_loss, dev_accuracy = model.evaluate([dev_sequences1, dev_sequences2], dev_labels_seq)
print(f'Development Loss: {dev_loss}')
print(f'Development Accuracy: {dev_accuracy}')
