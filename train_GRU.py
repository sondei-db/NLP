import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Concatenate
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder

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

tokenizer = Tokenizer()
all_texts = train_data.iloc[:, 0].tolist() + train_data.iloc[:, 1].tolist()
tokenizer.fit_on_texts(all_texts)

train_sequences1 = tokenizer.texts_to_sequences(train_data.iloc[:, 0])
train_sequences2 = tokenizer.texts_to_sequences(train_data.iloc[:, 1])
dev_sequences1 = tokenizer.texts_to_sequences(dev_data.iloc[:, 0])
dev_sequences2 = tokenizer.texts_to_sequences(dev_data.iloc[:, 1])
test_sequences1 = tokenizer.texts_to_sequences(test_data.iloc[:, 0])
test_sequences2 = tokenizer.texts_to_sequences(test_data.iloc[:, 1])


label_encoder = LabelEncoder()
all_labels = train_labels.iloc[:, 0].tolist() + dev_labels.iloc[:, 0].tolist()  # Add other label sets if necessary
label_encoder.fit(all_labels)

train_labels_seq = label_encoder.transform(train_labels.iloc[:, 0])
dev_labels_seq = label_encoder.transform(dev_labels.iloc[:, 0])

num_classes = len(label_encoder.classes_)

print("Starting model building...")
# Build the Model
input1 = Input(shape=(None,))
input2 = Input(shape=(None,))

embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100)
gru = GRU(100, return_sequences=False)
dense = Dense(100, activation='relu')

embedded1 = embedding(input1)
embedded2 = embedding(input2)

gru_output1 = gru(embedded1)
gru_output2 = gru(embedded2)

concatenated = Concatenate()([gru_output1, gru_output2])
dense_output = dense(concatenated)
output = Dense(num_classes, activation='softmax')(dense_output)

model = Model(inputs=[input1, input2], outputs=output)

print("Starting model compilation...")
# Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        print(f"Epoch {epoch+1} time: {self.times[-1]:.2f} seconds")


train_sequences1 = pad_sequences(train_sequences1)
train_sequences2 = pad_sequences(train_sequences2)
dev_sequences1 = pad_sequences(dev_sequences1)
dev_sequences2 = pad_sequences(dev_sequences2)

# Train the Model
print("Starting model training...")
time_callback = TimeHistory()
history = model.fit(
    [train_sequences1, train_sequences2],
    train_labels_seq,
    validation_data=([dev_sequences1, dev_sequences2], dev_labels_seq),
    epochs=1,
    batch_size=32,
    callbacks = [time_callback],
    verbose=1
)

# Save the Model
model.save('model.h5')

# Save Predictions to out.tsv
def save_predictions(sequences1, sequences2, label_encoder, file_path):
    sequences1 = pad_sequences(sequences1)
    sequences2 = pad_sequences(sequences2)
    predictions = model.predict([sequences1, sequences2])
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_words = label_encoder.inverse_transform(predicted_labels)
    df = pd.DataFrame(predicted_words, columns=["predicted_gap"])
    df.to_csv(file_path, sep='\t', index=False)

save_predictions(dev_sequences1, dev_sequences2, label_encoder, 'dev-0/out.tsv')
save_predictions(test_sequences1, test_sequences2, label_encoder, 'test-A/out.tsv')