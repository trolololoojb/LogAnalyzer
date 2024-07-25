"""
Test Modell. statt sigmoid und binary crossentropy jetzt tanh und mean squared error. Hoffnung, dass das gegen die 0 beim padding hilft.
Aber immer noch das Probölem, dass die hinteren variablen teile nicht erkannt werden.
"""


import csv
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint
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
logs_file_path = r"Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/content_list_bgl_unique.txt"
labels_file_path = r"Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/label_list_bgl_unique.csv"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
epochs = 10
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
logs = load_logs(logs_file_path)
labels = load_labels(labels_file_path)

# Laden der Daten
logs = []
labels = []

for logs_file_path, labels_file_path in zip(path.unique_content_path_list, path.unique_label_path_list):
    logs += load_logs(logs_file_path)
    labels += load_labels(labels_file_path)



#BGL Teil
tokenizer = BPE.generateTokenizer_BPE(logs, 10000)
tokenizer.save(directory_path + '/tokenizer.json')
sequences = [tokenizer.encode(log).ids for log in logs]
tokens = [tokenizer.encode(log).tokens for log in logs]
word_index =tokenizer.get_vocab()

#labels an subword encoding anpassen
labels_new = []
for sequence, label in zip(tokens, labels):
    labels_new.append(BPE.BPE_labels(sequence, label))

labels = labels_new

# Tokenisierung der Logeinträge
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(logs)
# sequences = tokenizer.texts_to_sequences(logs)
# word_index = tokenizer.word_index
# tokenizer_json = tokenizer.to_json()
# with open('tokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))


# Padding der Sequenzen
max_length = max(len(seq) for seq in sequences)
with open(directory_path + '/max_length.txt', 'w') as file:
    file.write(str(max_length))
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
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='tanh')))

# Kompilierung des Modells
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# Training des Modells
checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(dataset, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks= [checkpoint])


# Evaluation des Modells
loss, accuracy = model.evaluate(X_test, y_test)
with open(directory_path + '/training_results.txt', 'a') as file:
    file.write(f'Loss: {loss}, Accuracy: {accuracy}\n')
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    with open(directory_path + '/training_results.txt', 'a') as file:
        sequence = [tokenizer.encode(log).ids]
        sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        print(sequence_padded[0])
        file.write(str(sequence_padded[0]) + "\n")
        prediction = model.predict(sequence_padded)[0]
        
        #Nicht an BPE angepasste Labels:
        #words = log.split()
        
        #An BPE angepasste labels:
        words = tokenizer.encode(log).tokens
        result = []
        current_label = 'variabel' if prediction[0] > 0 else 'statisch'
        current_words = [words[0]]
        current_values = [prediction[0]]

        for word, pred in zip(words[1:], prediction[1:]):
            label = 'variabel' if pred > 0 else 'statisch'
            if label == current_label:
                current_words.append(word)
                current_values.append(pred)
            else:
                result.append((current_words, current_label, current_values))
                current_words = [word]
                current_values = [pred]
                current_label = label

        # Letzte Gruppe hinzufügen
        if current_words:
            result.append((current_words, current_label, current_values))

        # Ausgabe und Schreiben in Datei
        for group, label, values in result:
            text = ''.join(group)
            values_str = ', '.join(map(str, values))
            print(f'Wörter: {text}, Vorhersage: {label}, Werte: {values_str}')
            file.write(f'Wörter: {text}, Vorhersage: {label}, Werte: {values_str}\n')

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
    file.write("Training auf folgende Datei: " + logs_file_path + "\n")
    file.write(f'Total Trainings-Loss: {loss}\n')
    file.write(f'Total Trainings-Accuracy: {accuracy}\n')
    file.write(f'Total Validierungs-Loss: {val_loss}\n')
    file.write(f'Total Validierungs-Accuracy: {val_accuracy}\n')