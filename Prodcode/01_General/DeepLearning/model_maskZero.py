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
logs_file_path = path.twok_evaluate_content_list[0]
labels_file_path = path.twok_evaluate_label_list[0]

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
epochs = 10
log_examples_bgl = ["9 ddr errors(s) detected and corrected on rank 9, symbol 9, bit 9", "instruction cache parity error corrected", "total of 99 ddr error(s) detected and corrected"]
log_examples_hdfs = ["99.999.9.9:99999 Served block blk_-99999999999999999 to /99.999.9.9", "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_999999999999_9999/job.jar. blk_9999999999999999999", "99.999.99.999:99999 Starting thread to transfer block blk_-9999999999999999999 to 99.999.999.999:99999, 99.999.99.999:99999"]
log_examples_proxifier = ["rs.sinajs.cn:99 open through proxy proxy.cse.cuhk.edu.hk:9999 HTTPS", "pic9.zhimg.com:999 close, 9999 bytes (9.99 KB) sent, 9999 bytes (9.99 KB) received, lifetime 99:99", "ping9.teamviewer.com:999 (IPv9) error : Could not connect through proxy proxy.cse.cuhk.edu.hk:9999 - Proxy server cannot establish a connection with the target, status code 999"]
log_examples_hpc = ["Component State Change: Component \999alt9\999 is in the unavailable state (HWID=9999)", "unix.hw state_change.unavailable 9999999999 9 Component State Change: Component \999alt9\999 is in the unavailable state (HWID=9999)", "node-999 node status 9999999999 9 configured out"]
log_examples_zookeeper = ["My election bind port: 9.9.9.9/9.9.9.9:9999", "Closed socket connection for client /99.99.99.99:99999 which had sessionid 9x99f9a99999b999e", "caught end of stream exception"]

log_examples = [log_examples_bgl, log_examples_hdfs, log_examples_proxifier, log_examples_hpc, log_examples_zookeeper]

# Erstelle den gesamten Pfad
directory_path = f'Datensätze/Vorbereitete Daten/01_Models/{current_time}'
model_name = directory_path + '/tokenizedModel.keras'

# Stelle sicher, dass das Verzeichnis existiert
os.makedirs(directory_path, exist_ok=True)

# Laden der Daten
logs = load_logs(logs_file_path)
labels = load_labels(labels_file_path)

# Laden der Daten
logs = []
labels = []
for log_file_path, label_file_path in zip(path.unique_content_path_list, path.unique_label_path_list):
    logs += load_logs(log_file_path)
    labels += load_labels(label_file_path)

# Aufteilen in Trainings- und Testdaten
logs_train, logs_test, labels_train, labels_test = train_test_split(logs, labels, test_size=0.2, random_state=25)

with open(directory_path + '/validation_content.txt', 'w') as val_file:
    for line in logs_test:
        val_file.write(line +"\n")

with open(directory_path + '/validation_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for label in labels_test:
        writer.writerow(label)


# BPE Teil für Trainingsdaten
tokenizer = BPE.generateTokenizer_BPE(logs_train, 10000)
tokenizer.save(directory_path + '/tokenizer.json')
sequences_train = [tokenizer.encode(log).ids for log in logs_train]
tokens_train = [tokenizer.encode(log).tokens for log in logs_train]
word_index = tokenizer.get_vocab()

# Labels an subword encoding anpassen für Trainingsdaten
labels_train_new = []
for sequence, label in zip(tokens_train, labels_train):
    labels_train_new.append(BPE.BPE_labels(sequence, label))
labels_train = labels_train_new

# Padding der Sequenzen für Trainingsdaten
max_length = max(len(seq) for seq in sequences_train)
with open(directory_path + '/max_length.txt', 'w') as file:
    file.write(str(max_length))
sequences_train_padded = pad_sequences(sequences_train, maxlen=max_length, padding='post')
labels_train_padded = pad_sequences(labels_train, maxlen=max_length, padding='post')

# BPE Teil für Testdaten
sequences_test = [tokenizer.encode(log).ids for log in logs_test]
tokens_test = [tokenizer.encode(log).tokens for log in logs_test]

# Labels an subword encoding anpassen für Testdaten
labels_test_new = []
for sequence, label in zip(tokens_test, labels_test):
    labels_test_new.append(BPE.BPE_labels(sequence, label))
labels_test = labels_test_new

# Padding der Sequenzen für Testdaten
sequences_test_padded = pad_sequences(sequences_test, maxlen=max_length, padding='post')
labels_test_padded = pad_sequences(labels_test, maxlen=max_length, padding='post')


# Erstellung des Datensets für Trainingsdaten
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((sequences_train_padded, labels_train_padded))
dataset = dataset.shuffle(buffer_size=len(sequences_train_padded))
dataset = dataset.batch(batch_size)

# Erstellung des Modells
model = Sequential()
model.add(Embedding(input_dim=len(word_index), output_dim=64, input_length=max_length))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # Änderung der Aktivierungsfunktion zu sigmoid

# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Änderung der Verlustfunktion zu binary_crossentropy

# Training des Modells
checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(dataset, epochs=epochs, batch_size=batch_size, validation_data=(sequences_test_padded, labels_test_padded), callbacks=[checkpoint])

# Evaluation des Modells
loss, accuracy = model.evaluate(sequences_test_padded, labels_test_padded)

with open(directory_path + '/training_results.txt', 'a') as file:
    file.write(f'Loss: {loss}, Accuracy: {accuracy}\n')
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    with open(directory_path + '/training_results.txt', 'a') as file:
        sequence = [tokenizer.encode(log).ids]
        sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        print(sequence_padded[0])
        file.write("\nLog: " + log +"\n")
        file.write(str(sequence_padded[0]) + "\n")
        prediction = model.predict(sequence_padded)[0]
        
        #Nicht an BPE angepasste Labels:
        #words = log.split()
        
        #An BPE angepasste labels:
        words = tokenizer.encode(log).tokens
        result = []
        current_label = 'variabel' if prediction[0] > 0.5 else 'statisch'
        current_words = [words[0]]
        current_values = [prediction[0]]

        for word, pred in zip(words[1:], prediction[1:]):
            label = 'variabel' if pred > 0.5 else 'statisch'
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
    file.write("Training auf folgende Datei: " + str(logs_file_path) + "\n")
    file.write(f'Total Trainings-Loss: {loss}\n')
    file.write(f'Total Trainings-Accuracy: {accuracy}\n')
    file.write(f'Total Validierungs-Loss: {val_loss}\n')
    file.write(f'Total Validierungs-Accuracy: {val_accuracy}\n')
