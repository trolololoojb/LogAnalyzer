import pandas as pd

def json_to_python_list(input_path, chunksize=1):
    # Ein leeres Listenobjekt, um die Daten zu speichern
    python_list = []
    
    try:
        # JSON-Datei in Chunks lesen
        json_reader = pd.read_json(input_path, lines=True, chunksize=chunksize)
        
        # Iteriere über jeden Chunk
        for chunk in json_reader:
            # Erweitere die Python-Liste mit den Daten des Chunks
            python_list.extend(chunk.to_dict(orient='records'))
            print("Ein Chunk wurde verarbeitet und der Liste hinzugefügt.")
            
        print("Alle Daten wurden erfolgreich in die Python-Liste geladen.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

    return python_list

# Dateipfad definieren
input_json_path = r'Datensätze\Cadet\ta1-cadets-e3-official-1000_flatted.json'

# Funktion aufrufen und das Ergebnis speichern
result_list = json_to_python_list(input_json_path)

# Optional: Zeige die ersten 10 Elemente der Liste, um zu überprüfen, ob es funktioniert hat
print(result_list[:1])

