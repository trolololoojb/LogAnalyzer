import pandas as pd

def convert_csv_to_json_chunked(input_path, output_path, chunksize=1000):
    # Iterator für das Einlesen der CSV in Chunks
    csv_chunk_iterator = pd.read_csv(input_path, chunksize=chunksize)
    
    # Iteriere über jeden Chunk
    for i, chunk in enumerate(csv_chunk_iterator):
        # Konvertiere den aktuellen Chunk in JSON
        json_chunk = chunk.to_json(orient='records')
        
        # Bestimme den Namen der JSON-Datei für diesen Chunk
        chunk_output_path = f"{output_path[:-5]}-{i+1}.json"
        
        # Schreibe den JSON-Chunk in eine Datei
        with open(chunk_output_path, 'w') as file:
            file.write(json_chunk)
        print(f"Chunk {i+1} wurde gespeichert: {chunk_output_path}")

# Dateipfade definieren
input_csv_path = r'Datensätze\Cadet\ta1-cadets-e3-official-1000.csv'
output_json_path = r'Datensätze\Cadet\ta1-cadets-e3-official-1000.json'

# Funktion aufrufen
convert_csv_to_json_chunked(input_csv_path, output_json_path)

