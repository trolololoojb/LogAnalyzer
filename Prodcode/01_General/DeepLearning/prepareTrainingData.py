import csv
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import tqdm

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


    # Beispielhafte Logdaten
    # log_data = [
    #     "2023-07-15 12:34:56 ERROR Server failed to respond",
    #     "2023-07-15 12:35:01 INFO User login successful"
    # ]
    print(tokenizer.get_vocab())
    # # Tokenisierung der Logdaten
    # for log in log_data:
    #     encoded = tokenizer.encode(log)
    #     print(f"Original Log: {log}")
    #     print(f"Tokens: {encoded.tokens}")
    #     print(f"Token IDs: {encoded.ids}")
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
                    writer.writerow(sentence)
                
                processed_size += len(chunk.encode('utf-8'))  # Update processed size
                pbar.update(len(chunk.encode('utf-8')))

        # Verarbeitung des letzten leftovers, falls vorhanden
        if leftover:
            text = leftover.split()
            sentence = []
            for word in text:
                word_output = tokenizer.encode(word)
                sentence.append(word_output.ids)
            writer.writerow(sentence)

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

for log_file, tokenized_file in zip(content_file_path_list, tokenized_file_path_list):
    generateTokenizedData(tokenizer, log_file, tokenized_file, 10000)



#generateTokenizedData(tokenizer, r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\content_list_big.txt", r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.csv", 10000)
