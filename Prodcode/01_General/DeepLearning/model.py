import csv
import pandas as pd
import tensorflow as tf
from tokenizers import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime


def generator_data(data_path: str, batch_size: int, epochs: int):
    for e in range(epochs):
        gen = pd.read_csv(data_path, chunksize=batch_size)
        for df in gen:
            yield df[['x']].values, df['y'].values



# Beispiel X und Y Daten
X_data_files = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/content_list_bgl.txt',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/content_list_hdfs.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/content_list_hpc.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/content_list_proxifier.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/content_list_zookeeper.txt"
]
Y_data_files = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/label_list_bgl.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/label_list_zookeeper.csv"
]

unique_label_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/label_list_bgl_unique.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/unique_data/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/unique_data/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/unique_data/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/unique_data/label_list_zookeeper.csv"
]


unique_content_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/content_list_bgl_unique.txt',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/unique_data/content_list_hdfs.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/unique_data/content_list_hpc.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/unique_data/content_list_proxifier.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/unique_data/content_list_zookeeper.txt"
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


total_loss = []
total_accuracy = []
total_val_loss = []
total_val_accuracy = []
def model_train(logs_file_path, labels_file_path, model, tokenizer, max_length, checkpoint, early_stopping):
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
    history = model.fit(X_train, y_train, epochs=100, batch_size=1000, validation_data=(X_test, y_test), callbacks= [checkpoint, early_stopping])
    with open(f'Datensätze/Vorbereitete Daten - Beispiel/Models/{current_time}_{dimensions}/training_results.txt', 'a') as file:
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        file.write("Training auf folgende Datei: " + logs_file_path)
        file.write(f'Total Trainings-Loss: {loss}\n')
        file.write(f'Total Trainings-Accuracy: {accuracy}\n')
        file.write(f'Total Validierungs-Loss: {val_loss}\n')
        file.write(f'Total Validierungs-Accuracy: {val_accuracy}\n')
        
        
        
        





tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer_BPE.json")
word_index = tokenizer.get_vocab()
max_length = 102
dimensions = 50
# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=dimensions))
model.add(Bidirectional(GRU(dimensions, return_sequences=True)))
model.add(TimeDistributed(Dense(dimensions, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f'Datensätze/Vorbereitete Daten - Beispiel/Models/{current_time}_{dimensions}/tokenizedModel.keras'
checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)


# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

# for logs_file_path, labels_file_path in zip(X_data_files, Y_data_files):
#     model_train(logs_file_path, labels_file_path, model, tokenizer, max_length, checkpoint, early_stopping)

model_train(unique_content_path_list[0], unique_label_path_list[0], model, tokenizer, max_length, checkpoint, early_stopping)




