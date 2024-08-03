import csv
import os
import random
import path
import model_load


def select_random_lines(input_file, log_file, label_file, output_content, output_label, num_lines=2000):


    # Datei 2 einlesen und Zeilen in ein Set speichern
    with open(log_file, 'r') as f:
        datei2_inhalt = set(f.read().splitlines())

    # Datei 1 und 3 einlesen
    with open(input_file, 'r') as f:
        datei1_zeilen = f.read().splitlines()

    with open(label_file, 'r') as f:
        datei3_zeilen = f.read().splitlines()

    if len(datei1_zeilen) != len(datei3_zeilen):
        raise ValueError("Die Dateien 1 und 3 müssen die gleiche Anzahl an Zeilen haben.")

    # Ausgewählte Zeilen speichern
    ausgewaehlte_zeilen_input_file = []
    ausgewaehlte_zeilen_label_file = []
    counter = 0
    # Zufällige Auswahl und Überprüfung
    while len(ausgewaehlte_zeilen_input_file) < num_lines:
        print(len(ausgewaehlte_zeilen_input_file))

        zufaellige_index = random.randint(0, len(datei1_zeilen) - 1)
        zufaellige_zeile = datei1_zeilen[zufaellige_index]
        if counter <=1000:
            if zufaellige_zeile not in datei2_inhalt and zufaellige_zeile not in ausgewaehlte_zeilen_input_file:
                ausgewaehlte_zeilen_input_file.append(zufaellige_zeile)
                ausgewaehlte_zeilen_label_file.append(datei3_zeilen[zufaellige_index])
            else:
                counter += 1
        else:
            ausgewaehlte_zeilen_input_file.append(zufaellige_zeile)
            ausgewaehlte_zeilen_label_file.append(datei3_zeilen[zufaellige_index])
            counter = 0

    # Ergebnis in neuen Dateien speichern
    with open(output_content, 'w') as f:
        for zeile in ausgewaehlte_zeilen_input_file:
            f.write(zeile + '\n')

    with open(output_label, 'w') as f:
        for zeile in ausgewaehlte_zeilen_label_file:
            f.write(zeile + '\n')

    print(f'{num_lines} zufällige Zeilen wurden ausgewählt und gespeichert.')


def evaluate(eval_file, eval_label):
    file_name = os.path.basename(eval_file)
    model_name = os.path.basename(os.path.normpath(model_load.model_path))
    additional_infos = input(f"Es wird evaluiert: {file_name} mit dem Model {model_name}. Sonstige Infos hinzufügen:")
    with open(eval_file, 'r') as txt_f, open(eval_label, 'r') as csv_f:
        txt_zeilen = txt_f.readlines()
        csv_reader = csv.reader(csv_f)
        csv_zeilen = [list(map(int, row)) for row in csv_reader]
        
        if len(txt_zeilen) != len(csv_zeilen):
            raise ValueError("Die Dateien haben nicht die gleiche Anzahl an Zeilen")
        
        gleiche_zeilen = 0
        unterschiedliche_zeilen = 0
        
        for i in range(len(txt_zeilen)):
            print(i)
            pred_labels = model_load.predict_and_display(txt_zeilen[i].strip(), False)
            csv_ergebnis = csv_zeilen[i]
            if pred_labels == csv_ergebnis:
                gleiche_zeilen += 1
            else:
                unterschiedliche_zeilen += 1
                print(f"Zeile {i+1}: Unterschied gefunden")
                print(f"TXT: {pred_labels}")
                print(f"CSV: {csv_ergebnis}")
        
        gesamt_zeilen = len(txt_zeilen)
        gleiche_prozent = (gleiche_zeilen / gesamt_zeilen) * 100
        unterschiedliche_prozent = (unterschiedliche_zeilen / gesamt_zeilen) * 100
        
        with open(f"Datensätze/Vorbereitete Daten - Beispiel/03_Evaluationen/{file_name}_evaluation.txt", 'w') as infos:
            infos.write(f"Model: {model_name}\n Evaluierungsdaten: {file_name}\n Sonstige Infos: {additional_infos}\n")
            infos.write(f"Ergebnisse: \n    -Gleiche Zeilen: {gleiche_zeilen} ({gleiche_prozent:.2f}%)\n    -Unterschiedliche Zeilen: {unterschiedliche_zeilen} ({unterschiedliche_prozent:.2f}%)")
        print(f"Gleiche Zeilen: {gleiche_zeilen} ({gleiche_prozent:.2f}%)")
        print(f"Unterschiedliche Zeilen: {unterschiedliche_zeilen} ({unterschiedliche_prozent:.2f}%)")
    


# for input_file, log_file, label_file, output_content, output_label in zip(path.content_file_path_list, path.twok_log_path_list, path.label_list_path_list , path.twok_evaluate_content_list, path.twok_evaluate_label_list):
#     select_random_lines(input_file, log_file, label_file, output_content, output_label)

# evaluate(r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/2k/Proxifier_2k_evaluate_content.txt", r"Datensätze/Vorbereitete Daten - Beispiel/proxifier_v1/2k/Proxifier_2k_evaluate_label.csv")

for content, label in zip(path.twok_evaluate_content_list, path.twok_evaluate_label_list):
    evaluate(content, label)