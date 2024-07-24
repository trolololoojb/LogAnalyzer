import csv
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import BytePairEncoding as BPE
from datetime import datetime
import path

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
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
epochs = 15
log_examples_bgl = ["9 ddr errors(s) detected and corrected on rank 9, symbol 9, bit 9", "instruction cache parity error corrected", "total of 99 ddr error(s) detected and corrected"]
log_examples_hdfs = ["99.999.9.9:99999 Served block blk_-99999999999999999 to /99.999.9.9", "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_999999999999_9999/job.jar. blk_9999999999999999999", "99.999.99.999:99999 Starting thread to transfer block blk_-9999999999999999999 to 99.999.999.999:99999, 99.999.99.999:99999"]
log_examples_proxifier = ["rs.sinajs.cn:99 open through proxy proxy.cse.cuhk.edu.hk:9999 HTTPS", "pic9.zhimg.com:999 close, 9999 bytes (9.99 KB) sent, 9999 bytes (9.99 KB) received, lifetime 99:99", "ping9.teamviewer.com:999 (IPv9) error : Could not connect through proxy proxy.cse.cuhk.edu.hk:9999 - Proxy server cannot establish a connection with the target, status code 999"]

log_examples = [log_examples_bgl, log_examples_hdfs, log_examples_proxifier]

# Erstelle den gesamten Pfad
directory_path = f'Datensätze/Vorbereitete Daten - Beispiel/01_Models/{current_time}'
model_name = directory_path + '/tokenizedModel.keras'

# Stelle sicher, dass das Verzeichnis existiert
os.makedirs(directory_path, exist_ok=True)

# Laden der Daten
logs = []
labels = []

for logs_file_path, labels_file_path in zip(path.unique_content_path_list, path.unique_label_path_list):
    logs += load_logs(logs_file_path)
    labels += load_labels(labels_file_path)

print(len(logs))
print(len(labels))

# BPE-Teil
tokenizer = BPE.generateTokenizer_BPE(logs, 300)
tokenizer.save(directory_path + '/tokenizer.json')
sequences = [tokenizer.encode(log).ids for log in logs]
tokens = [tokenizer.encode(log).tokens for log in logs]
word_index = tokenizer.get_vocab()

# Padding der Sequenzen
max_length = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Umwandeln der Labels von -1 und 1 zu 0 und 1
labels_transformed = [[(label + 1) // 2 for label in sequence] for sequence in labels]

# One-hot-Encoding der Labels
num_classes = 2  # Annahme: 2 Klassen (statisch und variabel)
labels_onehot = [[to_categorical(label, num_classes=num_classes) for label in sequence] for sequence in labels_transformed]

# Padding der Labels
labels_padded = pad_sequences(labels_onehot, maxlen=max_length, padding='post', dtype='float32')

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels_padded, test_size=0.2, random_state=25)

batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=len(X_train))
dataset = dataset.batch(batch_size)

# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_length))
model.add(Bidirectional(GRU(100, return_sequences=True)))
model.add(Bidirectional(GRU(100, return_sequences=True)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))  # Softmax für mehrklassige Klassifikation

# Kompilierung des Modells
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training des Modells
checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(dataset, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Evaluation des Modells
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    with open(directory_path + '/training_results.txt', 'a') as file:
        sequence = [tokenizer.encode(log).ids]
        sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        print(sequence_padded[0])
        file.write(str(sequence_padded[0]) + "\n")
        prediction = model.predict(sequence_padded)[0]
        
        # An BPE angepasste Labels:
        words = tokenizer.encode(log).tokens
        for word, pred in zip(words, prediction):
            label = 'variabel' if pred[1] > 0.5 else 'statisch'
            print(f'Wort: {word}, Vorhersage: {label}, Wert: {pred}')
            file.write(str(f'Wort: {word}, Vorhersage: {label}, Wert: {pred}\n'))

# Beispielvorhersage
print("\nVorhersagen für neuen Logeintrag:")
for log_list in log_examples:
    for log in log_list:
        predict_and_display(log)

with open(directory_path + '/training_results.txt', 'a') as file:
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    file.write(f'Total Trainings-Loss: {loss}\n')
    file.write(f'Total Trainings-Accuracy: {accuracy}\n')
    file.write(f'Total Validierungs-Loss: {val_loss}\n')
    file.write(f'Total Validierungs-Accuracy: {val_accuracy}\n')
