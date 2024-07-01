import pandas as pd

#Wandelt eine strukturierte CSV Datei in zwei txt Dateien um. Eine davon ist die "content_list" welche die einzelnen Lognachrichten enthält. Die "label_list" enthält die Labels der einzelnen Wörter für die Lognachrichten. 
def process_log_data(file_path):
    try:
        # Lade die CSV-Datei
        data = pd.read_csv(file_path)
        
        # Extrahiere die notwendigen Spalten
        content_and_template = data[['Content', 'EventTemplate']]
        
        # Speichere die extrahierten Daten in einer neuen CSV-Datei
        content_and_template.to_csv('extracted_content_and_template.csv', index=False)
        
        # Erstelle die erste Liste mit Content
        content_list = content_and_template['Content'].tolist()
        
        # Erstelle die zweite Liste, die binäre Labels enthält
        binary_labels_list = []
        for index, row in content_and_template.iterrows():
            content_words = row['Content'].split()
            template_words = row['EventTemplate'].split()
            binary_labels = [0 if cw == tw else 1 for cw, tw in zip(content_words, template_words)]
            binary_labels_list.append(binary_labels)
        
        # Speichere beide Listen in einer Datei
        with open('content_list.txt', 'w') as file:
            file.write(str(content_list))

        
        with open('label_list.txt', 'w') as file:
            file.write(str(binary_labels_list))

        print("Verarbeitung abgeschlossen. Daten gespeichert.")
    except Exception as e:
        print("Ein Fehler ist aufgetreten:", e)

# Setze den Pfad zur CSV-Datei
file_path = r'Datensätze\HDFS\HDFS_2k.log_structured.csv'

# Funktion aufrufen
process_log_data(file_path)