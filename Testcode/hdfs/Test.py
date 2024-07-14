from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Erstellen eines BPE Tokenizers
tokenizer = Tokenizer(models.BPE())

# Definieren der PreTokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Definieren des Trainers
trainer = trainers.BpeTrainer(vocab_size=100000, special_tokens=["<pad>", "<cls>", "<sep>", "<unk>"])

# Pfad zur Logdatei
log_file_path = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Vorbereitete Daten - Beispiel\hdfs_v1\content_list_hdfs.txt"

# Training des Tokenizers
tokenizer.train(files=[log_file_path], trainer=trainer)


# Beispielhafte Logdaten
log_data = [
    "2023-07-15 12:34:56 ERROR Server failed to respond",
    "2023-07-15 12:35:01 INFO User login successful"
]

# Tokenisierung der Logdaten
for log in log_data:
    encoded = tokenizer.encode(log)
    print(f"Original Log: {log}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")