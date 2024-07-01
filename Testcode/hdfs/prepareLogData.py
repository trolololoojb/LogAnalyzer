# Wandelt eine Log Datei mit Hilfe einer CSv Datei mit Content Temnplates in zwei txt Dateien um. 
# Eine davon ist die "content_list" welche die einzelnen Lognachrichten enthält. 
# Die "label_list" enthält die Labels der einzelnen Wörter für die Lognachrichten. 
# Nicht Speicheroptimiert!!!!!

import pandas as pd
import re

#Wandelt die CSV Datei mit Templates in eine Liste um
def csvToList(csv_path):
    # Lade die CSV-Datei
    data = pd.read_csv(csv_path)

    event_templates = data['EventTemplate'].tolist()
    template_list = []
    for template in event_templates:
        # Create regex pattern from template
        regex_pattern = re.escape(template)
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
    comparison_result = []
    template_idx = 0
    for word in line_words:
        binary_labels = [0 if cw == tw else 1 for cw, tw in zip(line_words, template_words)]
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
    print("Labels wurden erstellt")
    return results



#Wandelt die hdfs Logdatei in eine Liste um. Entfernt den nicht benötigten Anfang jeder Zeile.
def process_log_file(log_file_path):
    processed_lines = []
    chunk_size = 5000

    with open(log_file_path, 'r') as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                # Find the position of the first occurrence of ": "
                pos = line.find(': ')
                if pos != -1:
                    # Remove the beginning of the line up to ": "
                    processed_line = line[pos + 2:]
                    processed_lines.append(processed_line.strip())
            chunk = file.readlines(chunk_size)
    print("Logdatei wurde in Liste umgewandelt")
    return processed_lines

# Beispielverwendung
log_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\HDFS\HDFS.log'
csv_file_path = r'C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\Datensätze\HDFS\unique_event_templates.csv'
list_templates = csvToList(csv_file_path)
list_logs = process_log_file(log_file_path)

processed_results = process_lines(list_logs, list_templates)

with open('label_list_big.txt', 'w') as file:
    file.write(str(processed_results))


with open('content_list_big.txt', 'w') as file:
    file.write(str(list_logs))


for line in processed_results[:5]:
    print(line)

for line in list_logs[:5]:
    print(line)