# Wandelt eine Log Datei mit Hilfe einer CSv Datei mit Content Temnplates in zwei txt Dateien um. 
# Eine davon ist die "content_list" welche die einzelnen Lognachrichten enthält. 
# Die "label_list" enthält die Labels der einzelnen Wörter für die Lognachrichten. 

import sys
import pandas as pd
import re
import csv
import os
import gc


def toList(file_path):
    """
    Bestimmt den Dateityp basierend auf der Dateiendung und gibt eine Liste der Einträge zurück.
    """
    # Extrahiere die Dateiendung
    _, file_extension = os.path.splitext(file_path)
    
    # Bestimme den Dateityp basierend auf der Dateiendung
    if file_extension.lower() == '.txt':
        return txtToList(file_path)
    elif file_extension.lower() == '.csv':
        return csvToList(file_path)

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




def match_template(line, templates):
    """
    Findet das passende Template für eine gegebene Zeile.
    """
    for template in templates:
        if re.search(template, line):
            return template
    return False

        




def compare_line_to_template(line, template):
    """
    Vergleicht eine Zeile mit einem Template und erstellt eine Liste von 0 und 1 für Übereinstimmung und Nicht-Übereinstimmung.
    """
    line = re.escape(line)
    line_words = line.split()
    template_words = template.split()
    padding = len(line_words) - len(template_words)
    if padding > 0:
        for i in range(padding):
            template_words.append(".*")
    comparison_result = []
    static_counter = 0
    for lw, tw in zip(line_words, template_words):
        binary_labels = 0 if lw == tw else 1 # Einzelne Wortübereinstimmungen in eine Liste packen
        if binary_labels == 1:
            static_counter +=1
        else:
            static_counter = 0
        comparison_result.append(binary_labels)
        if static_counter >= 7:
            return False
    return comparison_result



def process_lines(lines, templates):
    """ 
    Verarbeitet eine Liste von Zeilen und überprüft, welches Template auf jede Zeile zutrifft, und erstellt eine Vergleichsliste.
    """
    results = []
    filtered_lines = []
    for line in lines:
        matched_template = match_template(line, templates)
        if matched_template:
            comparison_result = compare_line_to_template(line, matched_template)
            if comparison_result:
                results.append(comparison_result)
                filtered_lines.append(line)
    return results, filtered_lines


def chunk_count(file_path, chunk_size):
    """
    Berechnet die Anzahl der Chunks basierend auf der Datei-Größe
    """
    with open(file_path, 'rb') as file:
        
        file_size = os.path.getsize(file_path)
        chunk_count = (file_size + chunk_size - 1) // chunk_size
    print(f"{chunk_count} Durchläufe benötigt")
    return chunk_count



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




def process_log_file(chunk, ident=1):
    """
    Wandelt die Logdatei in eine Liste um. Entfernt den nicht benötigten 
    Anfang jeder Zeile.
    
    Args:
    chunk: Teil der Logdatei
    ident: Datensatzidentifikator
    """
    processed_lines = []
    
    # Definiere die Funktion und Entferne Position basierend auf ident
    if ident == "hdfs":
        pos_finder = pos_finder_hdfs
        rem_pos = 2
    elif ident == "bgl":
        pos_finder = pos_finder_bgl
        rem_pos = 0  # Annahme, dass rem_pos von pos_finder_bgl zurückgegeben wird
    elif ident == "hpc":
        pos_finder = pos_finder_hpc
        rem_pos = 2


    for line in chunk:
        pos = pos_finder(line)
        if ident == "bgl":
            pos, rem_pos = pos
        
        if pos != -1:
            processed_line = line[pos + rem_pos:].strip()
            processed_lines.append(processed_line)
    
    return processed_lines


def pos_finder_hdfs(line):
    pos = line.find(': ')
    return pos

def pos_finder_bgl(line):
    level = ["INFO",
            "FATAL",
            "WARNING",
            "SEVERE",
            "ERROR"]
    for l in level:
        pos = line.find(l +' ')
        if pos != -1:
            return pos, len(l)
    return -1, -1

def pos_finder_hpc(line):
    pos = line.find('1 ')
    return pos

def process_start(log_file_path, csv_file_path, chunk_size):
    data_name = recognize_data()
    delete_file('content_list_'+ data_name + '.txt')
    delete_file('label_list_'+ data_name + '.csv')
    list_templates = toList(csv_file_path)
    chunkSize = chunk_count(log_file_path, chunk_size)
    chunkCount = 0
    leftover = ""

    with open(log_file_path, 'r', encoding='utf-8') as file:
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

            processed_lines = process_log_file(lines, data_name)
            
            del lines
            gc.collect()
            processed_results, processed_lines = process_lines(processed_lines, list_templates)
            
            with open('label_list_'+ data_name + '.csv', 'a', newline='') as label_file:
                writer = csv.writer(label_file)
                writer.writerows(processed_results)
            
            with open('content_list_'+ data_name + '.txt', 'a', encoding='utf-8') as content_file:
                for line in processed_lines:
                    content_file.write(line + '\n')
            # Manuelle Speicherfreigabe und Garbage Collection
            del processed_lines
            del processed_results
            gc.collect()  # Garbage Collector aufrufen
            chunkCount += 1
            print("Teil " + str(chunkCount) + " von " + str(chunkSize) + " verarbeitet", flush=True)

    # Verarbeite die letzte unvollständige Zeile, falls vorhanden
    if leftover:
        processed_lines = process_log_file([leftover], data_name)
        processed_results, processed_lines = process_lines(processed_lines, list_templates)
        with open('label_list_'+ data_name + '.csv', 'a', newline='') as label_file:
            writer = csv.writer(label_file)
            writer.writerows(processed_results)

        with open('content_list_'+ data_name + '.txt', 'a', encoding='utf-8') as content_file:
            for line in processed_lines:
                content_file.write(line + '\n')
        # Manuelle Speicherfreigabe und Garbage Collection
        del processed_lines
        del processed_results
        gc.collect()  # Garbage Collector aufrufen

def recognize_data():
    if "hdfs" in log_file_path.lower():
        return "hdfs"
    elif "bgl" in log_file_path.lower():
        return "bgl"
    elif "hpc" in log_file_path.lower():
        return "hpc"
    else:
        return None


log_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Drain3 Datensätze\HDFS\HDFS.log'
csv_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Datensätze\Drain3 Datensätze\HDFS\preprocessed\HDFS.log_templates.csv'
process_start(log_file_path, csv_file_path, 1000000)