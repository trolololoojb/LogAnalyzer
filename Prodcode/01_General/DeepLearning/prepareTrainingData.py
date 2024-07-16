import csv
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

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


def generateTokenizedData(tokenizer, input_filename, output_filename, chunk_size=1024):
    """
    Verarbeitet eine Textdatei zeilenweise mit einem gegebenen Tokenizer und speichert das Ergebnis in einer CSV-Datei.

    Args:
    - tokenizer: Ein Tokenizer-Objekt, das eine Methode 'encode' hat.
    - input_filename: Der Pfad zur Eingabedatei (.txt).
    - output_filename: Der Pfad zur Ausgabedatei (.csv).
    - chunk_size: Die Größe des Chunks in Bytes, der aus der Datei gelesen wird.
    """

    with open(input_filename, 'r', encoding='utf-8') as file, \
         open(output_filename, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        leftover = ''
        
        while True:
            # Lese einen Chunk aus der Datei
            chunk = file.read(chunk_size)
            if not chunk:
                break
            
            # Füge den übrig gebliebenen Teil der letzten Iteration hinzu
            chunk = leftover + chunk
            
            # Teile den Chunk in Zeilen
            lines = chunk.split('\n')
            
            # Der letzte Teil ist möglicherweise eine unvollständige Zeile
            leftover = lines.pop() if chunk[-1] != '\n' else ''
            
            # Verarbeite jede vollständige Zeile
            for line in lines:
                if line.strip():  # Ignoriere leere Zeilen
                    encoded_line = tokenizer.encode(line)
                    writer.writerow(encoded_line.tokens)  # Schreibe Tokens in die CSV-Datei

        # Verarbeite die letzte Zeile, wenn sie vorhanden ist
        if leftover:
            encoded_line = tokenizer.encode(leftover)
            writer.writerow(encoded_line.tokens)


generateTokenizer()
