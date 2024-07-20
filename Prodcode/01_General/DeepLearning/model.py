import csv
import pandas as pd
import tensorflow as tf
from tokenizers import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split


def generator_data(data_path: str, batch_size: int, epochs: int):
    for e in range(epochs):
        gen = pd.read_csv(data_path, chunksize=batch_size)
        for df in gen:
            yield df[['x']].values, df['y'].values



# Beispiel X und Y Daten
X_data_files = [
    r'/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/tokenized_list_bgl.csv',
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/tokenized_list_hdfs.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/tokenized_list_hpc.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/tokenized_list_proxifier.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/tokenized_list_zookeeper.csv"
]
Y_data_files = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/label_list_bgl.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/label_list_zookeeper.csv"
]

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

# row_counter = 0
# size  = data_size(filename)
# print("start")
# while row_counter != "finish":
#     loaded_data, row_counter = read_from_csv(filename, size, row_counter)
    


# Modell
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



def model_train(logs_file_path, labels_file_path, model, tokenizer, max_length):
    # Laden der Daten
    logs = load_logs(logs_file_path)
    labels = load_labels(labels_file_path)
    print("Lade "+logs_file_path)
    # Tokenisierung der Logeinträge
    sequences = tokenizer.encode_batch(logs)
    sequences_ids = [encoding.ids for encoding in sequences]

    # Padding der Sequenzen
    sequences_padded = pad_sequences(sequences_ids, maxlen=max_length, padding='post')
    labels_padded = pad_sequences(labels, maxlen=max_length, padding='post')

    # Aufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_padded, test_size=0.2, random_state=25)


    
    # Training des Modells
    model.fit(X_train, y_train, epochs=10, batch_size=1000, validation_split=0.2)

tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer_BPE.json")
word_index = tokenizer.get_vocab()
max_length = 102
# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=64))
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

for logs_file_path, labels_file_path in zip(X_data_files, Y_data_files):
    model_train(logs_file_path, labels_file_path, model, tokenizer, max_length)
model.save("Model.keras")

# # Evaluation des Modells
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# # Funktion zur Ausgabe der Vorhersagen
# def predict_and_display(log):
#     sequence = tokenizer.texts_to_sequences([log])
#     sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
#     prediction = model.predict(sequence_padded)[0]

#     words = log.split()
#     for word, pred in zip(words, prediction):
#         label = 'nicht statisch' if pred > 0.5 else 'statisch'
#         print(f'Wort: {word}, Vorhersage: {label}')

# # Beispielvorhersage
# new_log = "Received block blk_-1111111111111111111 of size 11111 from /11.111.11.111"
# print("\nVorhersagen für neuen Logeintrag:")
# predict_and_display(new_log)

