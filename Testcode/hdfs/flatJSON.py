import pandas as pd

def flatten_json(input_path, output_path, chunksize=100000):
    #Zähler
    i = 0

    # JSON-Datei in Chunks lesen
    json_reader = pd.read_json(input_path, lines=True, chunksize=chunksize)

    # Ausgabe vorbereiten
    first_chunk = True
    
    # Iteriere über jeden Chunk
    for chunk in json_reader:
        i+=1
        # Flatten the chunk using json_normalize
        flattened_chunk = pd.json_normalize(chunk.to_dict(orient='records'))
        
        # Schreibe den flachen Chunk in eine Datei, füge an, wenn es nicht der erste Chunk ist
        if first_chunk:
            flattened_chunk.to_json(output_path, orient='records', lines=True)
            first_chunk = False
        else:
            flattened_chunk.to_json(output_path, mode='a', orient='records', lines=True, header=False)
            
        print("Chunk {i} wurde verarbeitet und gespeichert.")

# Dateipfade definieren
input_json_path = r'Datensätze\Cadet\ta1-cadets-e3-official-1000.json'
output_flattened_json_path = r'Datensätze\Cadet\ta1-cadets-e3-official-1000_flatted.json'

# Funktion aufrufen
flatten_json(input_json_path, output_flattened_json_path)

