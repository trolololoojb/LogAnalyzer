import pandas as pd
import json

# Pfad zur JSON-Datei
json_file_path = r'Datensätze\Cadet\ta1-cadets-e3-official.json'
csv_file_path = r'Datensätze\Cadet\ta1-cadets-e3-official.csv'

# Anzahl der Zeilen pro Block
chunk_size = 2000

# Initialisiere eine leere Liste zur Speicherung der DataFrames
chunks = []
i = 0
# JSON-Datei zeilenweise einlesen und in Blöcken verarbeiten
with open(json_file_path, 'r', encoding='utf-8') as file:
    chunk = []
    for line in file:
        chunk.append(json.loads(line))
        if len(chunk) >= chunk_size:
            df_chunk = pd.json_normalize(chunk)
            chunks.append(df_chunk)
            i+=1
            print(i)
            chunk = []  # Leere den Block nach der Verarbeitung

    # Verarbeite den letzten Block, falls er nicht leer ist
    if chunk:
        df_chunk = pd.json_normalize(chunk)
        chunks.append(df_chunk)

# Verbinde alle DataFrames zu einem einzigen DataFrame
df = pd.concat(chunks, ignore_index=True)

# Speichere den gesamten DataFrame in eine CSV-Datei
df.to_csv(csv_file_path, index=False)

print(f'Die Datei wurde erfolgreich in {csv_file_path} umgewandelt.')

