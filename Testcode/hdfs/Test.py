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


import csv
import math

import numpy as np

np.set_printoptions(threshold=np.inf)

# Funktion zum Lesen der Liste aus CSV
import csv

def data_size(file_path):
    total_lines = sum(1 for line in open(file_path))
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


# Beispielaufrufe
filename = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\tokenized_list_bgl.csv'


# Daten aus CSV-Datei lesen
row_counter = 0
size  = data_size(filename)
print("start")
final_max_inner_length = 0
final_max_outer_length = 0
while row_counter != "finish":
    loaded_data, row_counter = read_from_csv(filename, size, row_counter)
    max_inner_length = max(max(len(inner) for inner in outer) for outer in loaded_data)
    if max_inner_length > final_max_inner_length:
        final_max_inner_length = max_inner_length
    max_outer_length = max(len(outer) for outer in loaded_data)
    if max_outer_length > final_max_outer_length:
        final_max_outer_length = max_outer_length





def pad_sequence(sequence, max_length):
    return sequence + [0] * (max_length - len(sequence))

def pad_outer_list(outer, max_inner_length, max_outer_length):
    padded_outer = [pad_sequence(inner, max_inner_length) for inner in outer]
    while len(padded_outer) < max_outer_length:
        padded_outer.append([0] * max_inner_length)
    return padded_outer

def pad_data(data, max_inner_length, max_outer_length):
    padded_data = [pad_outer_list(outer, max_inner_length, max_outer_length) for outer in data]
    return padded_data


padded_data = pad_data(loaded_data, max_inner_length, max_outer_length)


padded_data = np.array(padded_data)
print(padded_data)


















# -----------------------------------------------------------------------------

# import h5py
# import numpy as np

# # Daten definieren
# data = [[997, 113, 72, 166, 153, 16, 75, 70, 156, 105, 76, 57, 89, 28, 71, 709],
#  [75, 134],
#  [113],
#  [18]],
# [[997, 113, 72, 166, 153, 16, 72, 166, 153, 704, 480],
#  [75, 134],
#  [113],
#  [18]],
# [[44, 76, 145, 198],
#  [73, 77, 100, 296],
#  [761, 112]],
# [[161, 90, 45, 65, 106],
#  [75, 134],
#  [113],
#  [85]],
# [[170, 44, 956, 397, 45, 65, 106, 594],
#  [75, 134],
#  [113],
#  [15, 18]],
# [[333, 44, 956, 397, 45, 65, 106, 594],
#  [75, 134],
#  [113],
#  [15, 18]]

# # HDF5-Datei erstellen
# file_name = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.h5"
# with h5py.File(file_name, 'w') as hf:
#     # Daten in HDF5-Datei speichern
#     dataset = hf.create_dataset('data', data=np.array(data))

# print(f"Daten wurden in der Datei '{file_name}' gespeichert.")


# with h5py.File(file_name, 'r') as hf:
#     # Dataset aus der HDF5-Datei lesen
#     data = hf['data'][:]
    
# # Die gelesenen Daten anzeigen
# print("Gelesene Daten:")
# print(data)

