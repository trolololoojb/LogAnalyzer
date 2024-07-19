import csv
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import tqdm

tokenizer = Tokenizer.from_file(r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\Tokenizer\tokenizer.json")
content_file_path_list = [
    r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\content_list_bgl.txt',
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\content_list_hdfs.txt",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\content_list_hpc.txt",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\content_list_proxifier.txt",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\content_list_zookeeper.txt"
]

tokenized_file_path_list = [
    r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\tokenized_list_bgl.csv',
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\tokenized_list_hdfs.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\tokenized_list_hpc.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\tokenized_list_proxifier.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\tokenized_list_zookeeper.csv"
]

padded_file_path_list= [
    r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\padded_list_bgl.csv',
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\padded_list_hdfs.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\padded_list_hpc.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\padded_list_proxifier.csv",
    r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\padded_list_zookeeper.csv"
]

def generateTokenizer():
    # Erstellen eines BPE Tokenizers
    tokenizer = Tokenizer(models.BPE())

    # Definieren der PreTokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Definieren des Trainers
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<cls>", "<sep>", "<unk>"])

    # Pfad zur Logdatei
    log_file_path_hdfs = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\content_list_hdfs.txt"
    log_file_path_bgl = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\bgl_v1\content_list_bgl.txt"
    log_file_path_hpc = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hpc_v1\content_list_hpc.txt"
    log_file_path_proxifier = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\proxifier_v1\content_list_proxifier.txt"
    log_file_path_zookeeper = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\zookeeper_v1\content_list_zookeeper.txt"


    # Training des Tokenizers
    tokenizer.train(files=[log_file_path_hdfs, log_file_path_bgl, log_file_path_hpc, log_file_path_proxifier, log_file_path_zookeeper], trainer=trainer)

    print(tokenizer.get_vocab())
    tokenizer.save(r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\Tokenizer\tokenizer.json")


def delete_file(file_path):
    """
    Löscht die zu produzierenden Dateien um eine nicht gewollte Datenmanipulation zu vermeiden
    """
    try:
        os.remove(file_path)
        print(f"{file_path} wurde gelöscht.")
    except FileNotFoundError:
        print(f"{file_path} wurde nicht gefunden und konnte nicht gelöscht werden.")
    except Exception as e:
        print(f"Ein Fehler ist beim Löschen von {file_path} aufgetreten: {e}")


def generateTokenizedData(tokenizer, input_filename, output_filename, chunk_size=1024):
    """
    Verarbeitet eine Textdatei zeilenweise mit einem gegebenen Tokenizer und speichert die
    Token-IDs zusammen mit den Tokens in einer CSV-Datei. Enthält eine Fortschrittsanzeige.

    Args:
    - tokenizer: Ein Tokenizer-Objekt, das eine Methode 'encode' hat.
    - input_filename: Der Pfad zur Eingabedatei (.txt).
    - output_filename: Der Pfad zur Ausgabedatei (.csv).
    - chunk_size: Die Größe des Chunks in Bytes, der aus der Datei gelesen wird.
    """
    delete_file(output_filename)
    # Dateigröße bestimmen für die Fortschrittsanzeige
    total_size = os.path.getsize(input_filename)
    processed_size = 0

    with open(input_filename, 'r', encoding='utf-8') as file, \
         open(output_filename, 'a', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        leftover = ''
        
        # Fortschrittsanzeige initialisieren
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing File") as pbar:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunk = leftover + chunk  # Hinzufügen des leftovers zum aktuellen Chunk
                lines = chunk.split('\n')
                leftover = lines.pop()  # Entfernen und Speichern der unvollständigen letzten Zeile

                for line in lines:
                    text = line.split()
                    # Tokenisierung des Textes
                    sentence = []
                    for word in text:
                        word_output = tokenizer.encode(word)
                        sentence.append(word_output.ids)
                        writer.writerows(sentence)
                        writer.writerow([]) 
                
                processed_size += len(chunk.encode('utf-8'))  # Update processed size
                pbar.update(len(chunk.encode('utf-8')))

        # Verarbeitung des letzten leftovers, falls vorhanden
        if leftover:
            text = leftover.split()
            sentence = []
            for word in text:
                word_output = tokenizer.encode(word)
                sentence.append(word_output.ids)
                writer.writerows(sentence)
                writer.writerow([]) 



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
            if row_counter < row_count:
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

def save_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerows(row)
            writer.writerow([])  # Leerzeile zur Trennung der Matrizen



def get_max_padding_length():
    print("start max length process")
    final_max_inner_length = 0
    final_max_outer_length = 0
    for file_path in tokenized_file_path_list:
        row_counter = 0
        size  = data_size(file_path)
        while row_counter != "finish":
            loaded_data, row_counter = read_from_csv(file_path, size, row_counter, 10000000)
            max_inner_length = max(max(len(inner) for inner in outer) for outer in loaded_data)
            if max_inner_length > final_max_inner_length:
                final_max_inner_length = max_inner_length
            max_outer_length = max(len(outer) for outer in loaded_data)
            if max_outer_length > final_max_outer_length:
                final_max_outer_length = max_outer_length
                print(final_max_inner_length, final_max_outer_length)


def create_padding_files():
    final_max_inner_length = 212
    final_max_outer_length = 75

    print("start padding process")
    for file_path, save in zip(tokenized_file_path_list, padded_file_path_list):
        row_counter = 0
        delete_file(save)
        size  = data_size(file_path)
        while row_counter != "finish":
            loaded_data, row_counter = read_from_csv(file_path, size, row_counter, 1000)
            padded_data = pad_data(loaded_data, final_max_inner_length, final_max_outer_length)
            save_to_csv(padded_data, save)

create_padding_files()






# for log_file, tokenized_file in zip(content_file_path_list, tokenized_file_path_list):
#     generateTokenizedData(tokenizer, log_file, tokenized_file, 10000)



#generateTokenizedData(tokenizer, r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\content_list_big.txt", r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.csv", 10000)
