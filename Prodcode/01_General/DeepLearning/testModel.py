"""
Test Modell. statt sigmoid und binary crossentropy jetzt tanh und mean squared error. Hoffnung, dass das gegen die 0 beim padding hilft.
Aber immer noch das Probölem, dass die hinteren variablen teile nicht erkannt werden.
"""


import csv
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout, Masking
from sklearn.model_selection import train_test_split


# Laden der Logeinträge aus einer TXT-Datei
def load_logs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    return [log.strip() for log in logs]

# Laden der Labels aus einer CSV-Datei
def load_labels(file_path):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append([int(label) for label in row])
    return labels

# Pfade zu den Daten
logs_file_path = r"Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/content_list_bgl.txt"
labels_file_path = r"Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/label_list_bgl.csv"

# Laden der Daten
logs = load_logs(logs_file_path)
labels = load_labels(labels_file_path)

# Tokenisierung der Logeinträge
tokenizer = Tokenizer()
tokenizer.fit_on_texts(logs)
sequences = tokenizer.texts_to_sequences(logs)
word_index = tokenizer.word_index
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# Padding der Sequenzen
max_length = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
labels_padded = pad_sequences(labels, maxlen=max_length, padding='post')
# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_padded, test_size=0.2, random_state=25)

batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=len(X_train))
dataset = dataset.batch(batch_size)

# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=max_length))
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='tanh')))

# Kompilierung des Modells
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training des Modells
model.fit(dataset, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluation des Modells
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    sequence = tokenizer.texts_to_sequences([log])
    sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    print(sequence_padded[0])
    prediction = model.predict(sequence_padded)[0]
    

    words = log.split()
    for word, pred in zip(words, prediction):
        label = 'nicht statisch' if pred > 0.5 else 'statisch'
        print(f'Wort: {word}, Vorhersage: {label}')

# Beispielvorhersage
new_log = ["9 ddr errors(s) detected and corrected on rank 9, symbol 9, bit 9", "instruction cache parity error corrected", "total of 99 ddr error(s) detected and corrected"]
print("\nVorhersagen für neuen Logeintrag:")
for log in new_log:
    predict_and_display(log)