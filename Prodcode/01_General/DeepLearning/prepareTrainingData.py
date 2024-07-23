import csv
import os
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tokenizers.pre_tokenizers import Split

# Lade vorhandene Tokenizer
BPE_tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer_BPE.json")
tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer.json")

# Dateipfade für verschiedene Datensätze
content_file_path_list = [
    r'/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/content_list_bgl.txt',
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/content_list_hdfs.txt",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/content_list_hpc.txt",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/content_list_proxifier.txt",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/content_list_zookeeper.txt"
]

# Pfade für tokenisierte Dateien
tokenized_file_path_list = [
    r'/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/tokenized_list_bgl.csv',
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/tokenized_list_hdfs.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/tokenized_list_hpc.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/tokenized_list_proxifier.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/tokenized_list_zookeeper.csv"
]

# Pfade für BPE-tokenisierte Dateien
BPE_tokenized_file_path_list = [
    r'/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/BPE_tokenized_list_bgl.csv',
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/BPE_tokenized_list_hdfs.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/BPE_tokenized_list_hpc.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/BPE_tokenized_list_proxifier.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/BPE_tokenized_list_zookeeper.csv"
]

# Pfade für gepolsterte Dateien
padded_file_path_list= [
    r'/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/padded_list_bgl.csv',
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/padded_list_hdfs.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/padded_list_hpc.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/padded_list_proxifier.csv",
    r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/padded_list_zookeeper.csv"
]

# Pfade für Label-Dateien
label_list_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/label_list_bgl.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/label_list_zookeeper.csv"
]

# Pfade für eindeutige Labels
unique_label_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/label_list_bgl_unique.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/unique_data/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/unique_data/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/unique_data/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/unique_data/label_list_zookeeper.csv"
]

# Pfade für eindeutige Inhalte
unique_content_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/unique_data/content_list_bgl_unique.txt',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/unique_data/content_list_hdfs.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/unique_data/content_list_hpc.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/unique_data/content_list_proxifier.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/unique_data/content_list_zookeeper.txt"
]

def generateTokenizer_BPE():
    """
    Erstellt einen BPE Tokenizer und speichert ihn als JSON-Datei.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<cls>", "<sep>", "<unk>"])
    tokenizer.train(files=content_file_path_list, trainer=trainer)
    print(tokenizer.get_vocab())
    tokenizer.save(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer_BPE.json")

def generateTokenizer():
    """
    Erstellt einen WordLevel Tokenizer und speichert ihn als JSON-Datei.
    """
    tokenizer = Tokenizer(models.WordLevel())
    split_tokenizer = Split(pattern='r', behavior='removed', invert=True)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordLevelTrainer(min_frequency=1, special_tokens=["[PAD]", "[UNK]"])
    tokenizer.train(files=content_file_path_list, trainer=trainer)
    print(tokenizer.get_vocab())
    tokenizer.save(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer.json")

def delete_file(file_path):
    """
    Löscht eine Datei, falls sie existiert, um Datenmanipulation zu vermeiden.
    """
    try:
        os.remove(file_path)
        print(f"{file_path} wurde gelöscht.")
    except FileNotFoundError:
        print(f"{file_path} wurde nicht gefunden und konnte nicht gelöscht werden.")
    except Exception as e:
        print(f"Ein Fehler ist beim Löschen von {file_path} aufgetreten: {e}")

def generateTokenizedData_BPE(tokenizer, input_filename, output_filename, chunk_size=1024):
    """
    Tokenisiert eine Textdatei zeilenweise mit einem BPE Tokenizer und speichert die 
    Token-IDs in einer CSV-Datei.
    """
    delete_file(output_filename)
    total_size = os.path.getsize(input_filename)

    with open(input_filename, 'r', encoding='utf-8') as file, \
         open(output_filename, 'a', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        leftover = ''
        
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing File") as pbar:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunk = leftover + chunk
                lines = chunk.split('\n')
                leftover = lines.pop()

                for line in lines:
                    text = line.split()
                    sentence = []
                    for word in text:
                        word_output = tokenizer.encode(word)
                        sentence.append(word_output.ids)
                        writer.writerows(sentence)
                        writer.writerow([])
                
                pbar.update(len(chunk.encode('utf-8')))

        if leftover:
            text = leftover.split()
            sentence = []
            for word in text:
                word_output = tokenizer.encode(word)
                sentence.append(word_output.ids)
                writer.writerows(sentence)
                writer.writerow([])

def generate_tokenized_Data(tokenizer, input_filename, output_filename, chunk_size=1024):
    """
    Tokenisiert eine Textdatei zeilenweise mit einem WordLevel Tokenizer und speichert die 
    Token-IDs in einer CSV-Datei.
    """
    delete_file(output_filename)
    total_size = os.path.getsize(input_filename)

    with open(input_filename, 'r', encoding='utf-8') as file, \
         open(output_filename, 'a', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        leftover = ''
        
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing File") as pbar:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunk = leftover + chunk
                lines = chunk.split('\n')
                leftover = lines.pop()

                for line in lines:
                    output = tokenizer.encode(line)
                    writer.writerow(output.ids)
                
                pbar.update(len(chunk.encode('utf-8')))

        if leftover:
            output = tokenizer.encode(leftover)
            writer.writerow(output.ids)

def data_size(file_path):
    """
    Bestimmt die Anzahl der Zeilen in einer Datei.
    """
    total_lines = sum(1 for line in open(file_path))
    return total_lines

def read_from_csv_3d(filename, size, row_count=0, chunksize=10000000):
    """
    Liest eine CSV-Datei und gibt die Daten als 3D-Array zurück.
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        matrix = []
        chunksize = chunksize + row_count
        row_counter = 0
        for row in reader:
            if row_counter < row_count:
                row_counter += 1
                continue
            else:
                row_counter += 1
                if not row:
                    data.append(matrix)
                    matrix = []
                    if row_counter >= chunksize:
                        print(f"{row_counter} von {size} Zeilen verarbeitet")
                        return data, row_counter
                else:
                    matrix.append([int(num) for num in row])
        
        if matrix:
            data.append(matrix)
    
    return data, "finish"

def pad_sequence(sequence, max_length):
    """
    Füllt eine Sequenz mit Nullen auf die maximale Länge auf.
    """
    return sequence + [0] * (max_length - len(sequence))

def pad_outer_list(outer, max_inner_length, max_outer_length):
    """
    Füllt eine äußere Liste mit inneren Listen auf die maximale Länge auf.
    """
    padded_outer = [pad_sequence(inner, max_inner_length) for inner in outer]
    while len(padded_outer) < max_outer_length:
        padded_outer.append([0] * max_inner_length)
    return padded_outer

def pad_data(data, max_inner_length, max_outer_length):
    """
    Füllt die Daten auf die maximale Länge auf.
    """
    padded_data = [pad_outer_list(outer, max_inner_length, max_outer_length) for outer in data]
    return padded_data

def save_to_csv(data, filename):
    """
    Speichert die gepolsterten Daten in einer CSV-Datei.
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerows(row)
            writer.writerow([])

def get_max_padding_length():
    """
    Bestimmt die maximale Länge der Polsterung für die Daten.
    """
    print("start max length process")
    final_max_inner_length = 0
    final_max_outer_length = 0
    for file_path in BPE_tokenized_file_path_list:
        row_counter = 0
        size = data_size(file_path)
        while row_counter != "finish":
            loaded_data, row_counter = read_from_csv_3d(file_path, size, row_counter, 10000000)
            max_inner_length = max(max(len(inner) for inner in outer) for outer in loaded_data)
            if max_inner_length > final_max_inner_length:
                final_max_inner_length = max_inner_length
            max_outer_length = max(len(outer) for outer in loaded_data)
            if max_outer_length > final_max_outer_length:
                final_max_outer_length = max_outer_length
                print(final_max_inner_length, final_max_outer_length)

def get_max_length_tokenized():
    """
    Bestimmt die maximale Länge der tokenisierten Daten.
    """
    max_length = 0
    max_file = ""
    max_line = 0
    for file_path in tokenized_file_path_list:
        row_counter = 0
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                row_counter += 1
                row_length = len(row)
                if row_length > max_length:
                    max_length = row_length
                    max_line = row_counter
                    max_file = file_path
    print(max_file)
    print(max_line)
    print(max_length)
    return max_length

def delete_duplicates_with_labels(training_file, labels_file, output_training_file, output_labels_file):
    """
    Entfernt doppelte Einträge in den Trainingsdaten und den zugehörigen Labels.
    """
    try:
        with open(training_file, 'r') as tf:
            training_data = tf.readlines()

        with open(labels_file, newline='') as lf:
            reader = csv.reader(lf)
            labels_data = list(reader)

        if len(training_data) != len(labels_data):
            raise ValueError("Die Anzahl der Zeilen in den Trainingsdaten und Y-Daten stimmt nicht überein.")

        unique_training_data = []
        unique_labels_data = []
        seen = set()
        for i, line in enumerate(training_data):
            if line not in seen:
                seen.add(line)
                unique_training_data.append(line)
                unique_labels_data.append(labels_data[i])

        delete_file(output_training_file)
        delete_file(output_labels_file)

        with open(output_training_file, 'w') as tf_out:
            tf_out.writelines(unique_training_data)

        with open(output_labels_file, 'w', newline='') as lf_out:
            writer = csv.writer(lf_out)
            writer.writerows(unique_labels_data)

        print("Doppelte Einträge und die entsprechenden Y-Daten wurden entfernt.")
    except Exception as e:
        print(f"Fehler: {e}")

def create_padding_files():
    """
    Erstellt Dateien mit gepolsterten Daten basierend auf den maximalen inneren und äußeren Längen.
    """
    final_max_inner_length = 212
    final_max_outer_length = 75

    print("start padding process")
    for file_path, save in zip(tokenized_file_path_list, padded_file_path_list):
        row_counter = 0
        delete_file(save)
        size = data_size(file_path)
        while row_counter != "finish":
            loaded_data, row_counter = read_from_csv_3d(file_path, size, row_counter, 1000)
            padded_data = pad_data(loaded_data, final_max_inner_length, final_max_outer_length)
            save_to_csv(padded_data, save)


def BPE_labels(subwords, labels):
    new_labels = []
    counter = 0
    for subword in subwords:
        if subword != "#": 
            new_labels.append(labels[counter])
        else:
            new_labels.append(labels[counter])
            counter += 1
    return new_labels
    

# Aufruf der Methoden, falls erforderlich
# get_max_padding_length()
# create_padding_files()
# generateTokenizer()
# get_max_length_tokenized()

# Beispiel für die Tokenisierung und Entfernung von Duplikaten
# for log_file, tokenized_file in zip(content_file_path_list, BPE_tokenized_file_path_list):
#     generateTokenizedData_BPE(tokenizer, log_file, tokenized_file, 10000)

# for log_file, tokenized_file in zip(content_file_path_list, tokenized_file_path_list):
#     generate_tokenized_Data(tokenizer, log_file, tokenized_file, 10000)

for content_file, label_file, unique_content_file, unique_label_file in zip(content_file_path_list, label_list_path_list, unique_content_path_list, unique_label_path_list):
    delete_duplicates_with_labels(content_file, label_file, unique_content_file, unique_label_file)
