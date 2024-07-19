import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense
import tqdm


def generator_data(data_path: str, batch_size: int, epochs: int):
    for e in range(epochs):
        gen = pd.read_csv(data_path, chunksize=batch_size)
        for df in gen:
            yield df[['x']].values, df['y'].values


import numpy as np

# Beispiel X und Y Daten
X_data_files = [
    r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\tokenized_list_bgl.csv',
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\tokenized_list_hdfs.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\tokenized_list_hpc.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\tokenized_list_proxifier.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\tokenized_list_zookeeper.csv"
]
Y_data_files = [
    r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\label_list_bgl.csv',
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\label_list_hdfs.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\label_list_hpc.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\label_list_proxifier.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\label_list_zookeeper.csv"
]

def data_size(file_path):
    total_lines = sum(1 for line in open(filename))
    return total_lines


def read_from_csv(filename, size, row_count = 0, chunksize = 10000000):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        matrix = []
        chunksize = chunksize + row_count
        row_counter = 0
        for row in reader:
            if row_counter <= row_count:
                row_counter += 1
                continue
            else:
                row_counter += 1
                if not row:  # Leerzeile gefunden (Trennung der Matrizen)
                    
                    data.append(matrix)
                    matrix = []
                    if row_counter >= chunksize:
                        print(f"{row_counter} von {size} Zeilen verarbeitet")
                        return data, row_counter
                else:
                    matrix.append([int(num) for num in row])
        
        if matrix:  # Letzte Matrix hinzufügen, falls nicht leer
            data.append(matrix)
    
    return data, "finish"

row_counter = 0
size  = data_size(filename)
print("start")
while row_counter != "finish":
    loaded_data, row_counter = read_from_csv(filename, size, row_counter)
    
# Pad the sequences to ensure uniform input length
X_data_padded = tf.keras.preprocessing.sequence.pad_sequences(read_from_csv(r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.csv"), padding='post')
np.set_printoptions(threshold=np.inf)
print(X_data_padded)
# # Parameter
# vocab_size = 1001  # Vokabulargröße angepasst für Padding
# embedding_dim = 64  # Dimension der Embedding-Schicht
# max_length = len(X_data_padded[0])  # Maximale Länge der Eingabesequenzen

# # Modell
# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
#     Bidirectional(GRU(64)),
#     Dense(1, activation='sigmoid')
# ])

# # Modellkompilierung
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Modellzusammenfassung
# model.summary()




# gen = generator_data('data.csv', batch_size=2, epochs=200)
# next(gen)

# # Training
# X_data_padded = np.array(X_data_padded)
# Y_data = np.array(Y_data)
# model.fit(X_data_padded, Y_data, epochs=10, batch_size=2)

# # Neue Eingabedaten
# new_X_data = [[922, 16, 85, 19, 726], [948, 135, 738], [33, 45]]

# # Sequenzen auf dieselbe Länge wie Trainingsdaten padden
# new_X_data_padded = tf.keras.preprocessing.sequence.pad_sequences(new_X_data, padding='post', maxlen=max_length)

# # Modell zur Vorhersage verwenden
# predictions = model.predict(new_X_data_padded)

# # Ausgabe der Vorhersagen
# print("Vorhersagen:")
# for i, pred in enumerate(predictions):
#     print(f"Sequenz {i + 1}: {pred[0]:.4f} ({'positiv' if pred[0] >= 0.5 else 'negativ'})")