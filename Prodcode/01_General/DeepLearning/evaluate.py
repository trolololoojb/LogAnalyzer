import random
import path



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


def evaluate(eval_file, original_content, original_label):



for input_file, log_file, label_file, output_content, output_label in zip(path.content_file_path_list, path.twok_log_path_list, path.label_list_path_list , path.twok_evaluate_content_list, path.twok_evaluate_label_list):
    select_random_lines(input_file, log_file, label_file, output_content, output_label)


