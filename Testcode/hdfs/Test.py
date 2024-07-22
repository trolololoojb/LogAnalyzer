# import numpy as np
# import tensorflow as tf
# import csv


# # Pfad zur CSV-Datei
# file_path = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.csv"


# np.set_printoptions(threshold=np.inf)
# X_data = []

# # Angenommen, die Datei 'output.csv' existiert und enthält Daten im CSV-Format
# with open(file_path, 'r', newline='', encoding='utf-8') as file:
#     reader = csv.reader(file, delimiter= ";")
#     for row in reader:
#         # Konvertiere jede Zahl von String zu Integer
#         #converted_row = [int(num) for num in row]
#         X_data.append(row)


# print(X_data)

# X_data_padded = tf.keras.preprocessing.sequence.pad_sequences(X_data, padding='post')
# print(X_data_padded)


#-------------------------------------------------------------------------------


# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Beispielhafte Logdaten generieren
# logs = [
#     "INFO User logged in",
#     "ERROR Failed to connect to database",
#     "WARN Low disk space",
#     "INFO User logged out",
#     "ERROR Unexpected error occurred"
# ]

# # Labels für die Logtypen
# labels = ["INFO", "ERROR", "WARN"]

# # Daten vorbereiten
# X = np.array(logs)
# y = np.array([0, 1, 2, 0, 1])  # Indizes der Labels

# # Label-Encoding der Zielvariablen
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Tokenisierung und Padding der Sequenzen
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# X_tokenized = tokenizer.texts_to_sequences(X)
# X_padded = pad_sequences(X_tokenized, padding='post')

# # Train-Test-Split
# X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2)

# # Modell definieren
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_padded.shape[1]))
# model.add(LSTM(100))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(len(labels), activation='softmax'))

# # Modell kompilieren
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Modell trainieren
# model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# # Modell evaluieren
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # Beispielvorhersage
# new_logs = ["INFO User login attempt", "ERROR Disk failure"]
# new_logs_tokenized = tokenizer.texts_to_sequences(new_logs)
# new_logs_padded = pad_sequences(new_logs_tokenized, padding='post', maxlen=X_padded.shape[1])

# predictions = model.predict(new_logs_padded)
# predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# print(predicted_labels)





#----------------------------------------------------------------------------------------------------------------


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
logs_file_path = r"Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/content_list_bgl_unique.txt"
labels_file_path = r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/label_list_bgl_unique.csv"

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
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_padded, test_size=0.2)

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
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

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












