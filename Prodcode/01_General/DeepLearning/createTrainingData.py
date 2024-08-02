# Wandelt eine Log Datei mit Hilfe einer CSv Datei mit Content Temnplates in zwei txt Dateien um. 
# Eine davon ist die "content_list" welche die einzelnen Lognachrichten enthält. 
# Die "label_list" enthält die Labels der einzelnen Wörter für die Lognachrichten. 

import path
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
        regex_pattern = re.sub(r'\d', '9', regex_pattern)
        template_list.append(regex_pattern)
    print("Templates wurden in Liste umgewandelt")
    return template_list

def txtToList(txt_path):
    template_list = []

    # Öffne die TXT-Datei und lese sie Zeile für Zeile
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Ersetze Platzhalter durch reguläre Ausdrücke
            line = line.strip()
            regex_pattern = re.escape(line)
            regex_pattern = regex_pattern.replace(r'<\*>', r'.*')
            regex_pattern = re.sub(r'\d', '9', regex_pattern)
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

    filtered_templates= [item for item in template_words if '.*\\' not in item]
    comparison_result = []
    static_counter = 0
    padding_bool = True

    for lw in line_words:
        binary_labels = -1 if lw == filtered_templates[0] else 1 # Einzelne Wortübereinstimmungen in eine Liste packen
        if binary_labels == 1:
            static_counter +=1
        else:
            if padding_bool:
                del filtered_templates[:1]
            if len(filtered_templates) == 0:
                padding_bool == False
                filtered_templates.append("")
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
    elif ident == "proxifier":
        pos_finder = pos_finder_proxifier
        rem_pos = 2
    elif ident == "zookeeper":
        pos_finder = pos_finder_zookeeper
        rem_pos = 4


    for line in chunk:
        pos = pos_finder(line)
        if ident == "bgl":
            pos, rem_pos = pos
        
        if pos != -1:
            processed_line = line[pos + rem_pos:].strip()
            processed_line = re.sub(r'\d', '9', processed_line)
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

def pos_finder_zookeeper(line):
    pos = line.find('] - ')
    return pos

def pos_finder_proxifier(line):
    pos = line.find('- ')
    return pos

def process_start(log_file_path, csv_file_path, chunk_size, content_file_path, label_list_path):
    data_name = recognize_data(log_file_path)
    print("\nVerarbeitung des " + data_name +"-Datensatzes")
    delete_file(content_file_path)
    delete_file(label_list_path)
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
            
            with open(label_list_path, 'a', newline='') as label_file:
                writer = csv.writer(label_file)
                writer.writerows(processed_results)
            
            with open(content_file_path, 'a', encoding='utf-8') as content_file:
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
        with open(label_list_path, 'a', newline='') as label_file:
            writer = csv.writer(label_file)
            writer.writerows(processed_results)

        with open(content_file_path, 'a', encoding='utf-8') as content_file:
            for line in processed_lines:
                content_file.write(line + '\n')
        # Manuelle Speicherfreigabe und Garbage Collection
        del processed_lines
        del processed_results
        gc.collect()  # Garbage Collector aufrufen

def recognize_data(log_file_path):
    if "hdfs" in log_file_path.lower():
        return "hdfs"
    elif "bgl" in log_file_path.lower():
        return "bgl"
    elif "hpc" in log_file_path.lower():
        return "hpc"
    elif "proxifier" in log_file_path.lower():
        return "proxifier"
    elif "zookeeper" in log_file_path.lower():
        return "zookeeper"



log_file_path_list = [
    r"Datensätze/Drain3 Datensätze/BGL/BGL.log",
    r"Datensätze/Drain3 Datensätze/HDFS/HDFS.log",
    r"Datensätze/Drain3 Datensätze/HPC/HPC.log",
    r"Datensätze/Drain3 Datensätze/Proxifier/Proxifier.log",
    r'Datensätze/Drain3 Datensätze/Zookeeper/Zookeeper.log'
]

csv_file_path_list = [
    r"Datensätze/Drain3 Datensätze/BGL/BGL_templates.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/HDFS_v1_unique_event_templates.csv",
    r"Datensätze/Drain3 Datensätze/HPC/HPC_2k.log_templates.csv",
    r"Datensätze/Drain3 Datensätze/Proxifier/Proxifier_2k.log_templates.csv",
    r'Datensätze/Drain3 Datensätze/Zookeeper/Zookeeper_2k.log_templates.csv'
]

content_file_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/content_list_bgl.txt',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/content_list_hdfs.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/content_list_hpc.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/content_list_proxifier.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/content_list_zookeeper.txt"
]

new_content_file_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/content_list_bgl_new.txt',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/content_list_hdfs_new.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/content_list_hpc_new.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/content_list_proxifier_new.txt",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/content_list_zookeeper_new.txt"
]

label_list_path_list = [
    r'Datensätze/Vorbereitete Daten - Beispiel/bgl_v1/label_list_bgl.csv',
    r"Datensätze/Vorbereitete Daten - Beispiel/hdfs_v1/label_list_hdfs.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/hpc_v1/label_list_hpc.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/label_list_proxifier.csv",
    r"Datensätze/Vorbereitete Daten - Beispiel/zookeeper_v1/label_list_zookeeper.csv"
]

# for log_file_path, csv_file_path, content_file_path, label_list_path in zip(log_file_path_list, csv_file_path_list, content_file_path_list, label_list_path_list):
#     process_start(log_file_path, csv_file_path, 1000000, content_file_path, label_list_path)

#process_start(log_file_path_list[0], csv_file_path_list[0], 1000000, content_file_path_list[0], label_list_path_list[0])

for log_file_path, csv_file_path, content_file_path, label_list_path in zip(path.twok_log_path_list, path.twok_strctured_path_list, path.twok_content_path_list, path.twok_label_path_list):
    process_start(log_file_path, csv_file_path, 1000000, content_file_path, label_list_path)