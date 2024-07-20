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


# import csv
# import math

# import numpy as np

# np.set_printoptions(threshold=np.inf)

# # Funktion zum Lesen der Liste aus CSV
# import csv

# def data_size(file_path):
#     total_lines = sum(1 for line in open(file_path))
#     return total_lines


# def read_from_csv(filename, size, row_count = 0, chunksize = 10000000):
#     with open(filename, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         data = []
#         matrix = []
#         chunksize = chunksize + row_count
#         row_counter = 0
#         for row in reader:
#             if row_counter <= row_count:
#                 row_counter += 1
#                 continue
#             else:
#                 row_counter += 1
#                 if not row:  # Leerzeile gefunden (Trennung der Matrizen)
                    
#                     data.append(matrix)
#                     matrix = []
#                     if row_counter >= chunksize:
#                         print(f"{row_counter} von {size} Zeilen verarbeitet")
#                         return data, row_counter
#                 else:
#                     matrix.append([int(num) for num in row])
        
#         if matrix:  # Letzte Matrix hinzufügen, falls nicht leer
#             data.append(matrix)
    
#     return data, "finish"


# # Beispielaufrufe
# filename = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\tokenized_list_bgl.csv'


# # Daten aus CSV-Datei lesen
# row_counter = 0
# size  = data_size(filename)
# print("start")
# final_max_inner_length = 0
# final_max_outer_length = 0
# while row_counter != "finish":
#     loaded_data, row_counter = read_from_csv(filename, size, row_counter)
#     max_inner_length = max(max(len(inner) for inner in outer) for outer in loaded_data)
#     if max_inner_length > final_max_inner_length:
#         final_max_inner_length = max_inner_length
#     max_outer_length = max(len(outer) for outer in loaded_data)
#     if max_outer_length > final_max_outer_length:
#         final_max_outer_length = max_outer_length





# def pad_sequence(sequence, max_length):
#     return sequence + [0] * (max_length - len(sequence))

# def pad_outer_list(outer, max_inner_length, max_outer_length):
#     padded_outer = [pad_sequence(inner, max_inner_length) for inner in outer]
#     while len(padded_outer) < max_outer_length:
#         padded_outer.append([0] * max_inner_length)
#     return padded_outer

# def pad_data(data, max_inner_length, max_outer_length):
#     padded_data = [pad_outer_list(outer, max_inner_length, max_outer_length) for outer in data]
#     return padded_data


# padded_data = pad_data(loaded_data, max_inner_length, max_outer_length)


# padded_data = np.array(padded_data)
# print(padded_data)




#----------------------------------------------------------------------------------------------------------------


import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
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
logs_file_path = r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/content_list_hdfs.txt"
labels_file_path = r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/label_list_hdfs.csv"

# Laden der Daten
logs = load_logs(logs_file_path)
labels = load_labels(labels_file_path)

# Tokenisierung der Logeinträge
tokenizer = Tokenizer()
tokenizer.fit_on_texts(logs)
sequences = tokenizer.texts_to_sequences(logs)
word_index = tokenizer.word_index

# Padding der Sequenzen
max_length = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
labels_padded = pad_sequences(labels, maxlen=max_length, padding='post')

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_padded, test_size=0.2)

# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

# Evaluation des Modells
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    sequence = tokenizer.texts_to_sequences([log])
    sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(sequence_padded)[0]

    words = log.split()
    for word, pred in zip(words, prediction):
        label = 'nicht statisch' if pred > 0.5 else 'statisch'
        print(f'Wort: {word}, Vorhersage: {label}')

# Beispielvorhersage
new_log = "Received block blk_-1111111111111111111 of size 11111 from /11.111.11.111"
print("\nVorhersagen für neuen Logeintrag:")
predict_and_display(new_log)












