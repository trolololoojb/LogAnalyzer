# Wandelt eine Log Datei mit Hilfe einer CSv Datei mit Content Temnplates in zwei txt Dateien um. 
# Eine davon ist die "content_list" welche die einzelnen Lognachrichten enthält. 
# Die "label_list" enthält die Labels der einzelnen Wörter für die Lognachrichten. 
# Sepicheroptimiert

import pandas as pd
import re
import csv
import os
import gc

#Bestimmt den Dateityp basierend auf der Dateiendung.
def toList(file_path):
    # Extrahiere die Dateiendung
    _, file_extension = os.path.splitext(file_path)
    
    # Bestimme den Dateityp basierend auf der Dateiendung
    if file_extension.lower() == '.txt':
        return txtToList(file_path)
    elif file_extension.lower() == '.csv':
        return csvToList(file_path)


#Wandelt die CSV Datei mit Templates in eine Liste um
def csvToList(csv_path):
    # Lade die CSV-Datei
    data = pd.read_csv(csv_path)

    event_templates = data['EventTemplate'].tolist()
    template_list = []
    for template in event_templates:
        regex_pattern = re.escape(template)
        regex_pattern = regex_pattern.replace(r'<\*>', r'.*')
        template_list.append(regex_pattern)
    print("Templates wurden in Liste umgewandelt")
    return template_list

def txtToList(txt_path):
    template_list = []

    # Öffne die TXT-Datei und lese sie Zeile für Zeile
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Ersetze Platzhalter durch reguläre Ausdrücke
            line = line.replace('\n', r'')
            regex_pattern = re.escape(line)
            regex_pattern = regex_pattern.replace(r'<\*>', r'.*')
            template_list.append(regex_pattern)
    print("Templates wurden in Liste umgewandelt")
    return template_list



#Findet das passende Template für eine gegebene Zeile.
def match_template(line, templates, counter):

    for template in templates:
        if re.search(template, line):
            return template
    raise Exception("Template ohne Match!", templates, "Line:", counter,line)

        



#Vergleicht eine Zeile mit einem Template und erstellt eine Liste von 0 und 1 für Übereinstimmung und Nicht-Übereinstimmung.
def compare_line_to_template(line, template):
    
    line = re.escape(line)
    line_words = line.split()
    template_words = template.split()
    padding = len(line_words) - len(template_words)
    if padding > 0:
        for i in range(padding):
            template_words.append(".*")
    comparison_result = []
    for lw, tw in zip(line_words, template_words):
        binary_labels = 0 if lw == tw else 1 # Einzelne Wortübereinstimmungen in eine Liste packen
        comparison_result.append(binary_labels)
    return comparison_result


#Verarbeitet eine Liste von Zeilen und überprüft, welche Template auf jede Zeile zutrifft, und erstellt eine Vergleichsliste.
def process_lines(lines, templates):
    results = []
    counter = 0
    for line in lines:
        counter+=1
        matched_template = match_template(line, templates, counter)
        if matched_template:
            comparison_result = compare_line_to_template(line, matched_template)
            results.append(comparison_result)
    return results


def chunk_count(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        # Berechnet die Anzahl der Chunks basierend auf der Datei-Größe
        file_size = os.path.getsize(file_path)
        chunk_count = (file_size + chunk_size - 1) // chunk_size
    print(f"{chunk_count} Durchläufe benötigt")
    return chunk_count


# Löscht die zu produzierenden Dateien um eine nicht gewollte Datenmanipulation zu vermeiden
def delete_file(file_path):

    try:
        os.remove(file_path)
        print(f"{file_path} wurde gelöscht.")
    except FileNotFoundError:
        print(f"{file_path} wurde nicht gefunden und konnte nicht gelöscht werden.")
    except Exception as e:
        print(f"Ein Fehler ist beim Löschen von {file_path} aufgetreten: {e}")



#Wandelt die hdfs Logdatei in eine Liste um. Entfernt den nicht benötigten Anfang jeder Zeile.
def process_log_file(chunk):
    processed_lines = []

    for line in chunk:
        # Find the position of the first occurrence of ": "
        pos = line.find(': ')
        if pos != -1:
            # Remove the beginning of the line up to ": "
            processed_line = line[pos + 2:]
            processed_lines.append(processed_line.strip())
    return processed_lines

def process_start(log_file_path, csv_file_path, chunk_size):
    delete_file('content_list_big.csv')
    delete_file('label_list_big.csv')
    list_templates = toList(csv_file_path)
    chunkSize = chunk_count(log_file_path, chunk_size)
    chunkCount = 0
    leftover = ""

    with open(log_file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            # Füge übrig gebliebenen Teil von vorherigem Chunk hinzu
            chunk = leftover + chunk
            lines = chunk.splitlines()

            # Überprüfe, ob letzte Zeile vollständig ist
            if chunk[-1] != '\n':
                leftover = lines.pop()  # Speichere unvollständige Zeile
            else:
                leftover = ""

            processed_lines = process_log_file(lines)
            with open('content_list_big.txt', 'a', encoding='utf-8') as content_file:
                for line in processed_lines:
                    content_file.write(line + '\n')
            processed_results = process_lines(processed_lines, list_templates)
            with open('label_list_big.csv', 'a', newline='') as label_file:
                writer = csv.writer(label_file)
                writer.writerows(processed_results)
            # Manuelle Speicherfreigabe und Garbage Collection
            del processed_lines
            del processed_results
            gc.collect()  # Garbage Collector aufrufen
            chunkCount += 1
            print("Teil " + str(chunkCount) + " von " + str(chunkSize) + " verarbeitet", flush=True)

    # Verarbeite die letzte unvollständige Zeile, falls vorhanden
    if leftover:
        processed_lines = process_log_file([leftover])
        with open('content_list_big.csv', 'a', newline='') as content_file:
            writer = csv.writer(content_file)
            writer.writerows([[line] for line in processed_lines])
        processed_results = process_lines(processed_lines, list_templates)
        with open('label_list_big.csv', 'a', newline='') as label_file:
            writer = csv.writer(label_file)
            writer.writerows(processed_results)
        # Manuelle Speicherfreigabe und Garbage Collection
        del processed_lines
        del processed_results
        gc.collect()  # Garbage Collector aufrufen




log_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\Drain3 Datensätze\BGL\BGL.log'
csv_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\Drain3 Datensätze\BGL\BGL_templates.csv'
process_start(log_file_path, csv_file_path, 1000000)