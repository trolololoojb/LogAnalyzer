import csv
import os
import random
import path
import model_load
from sklearn.metrics import f1_score

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

def pad_labels(true_labels, predicted_labels):
    """
    Füllt die true_labels und predicted_labels auf, um die gleiche Länge zu erreichen.
    Dabei werden die Werte aus der jeweils anderen Liste mit negiertem Vorzeichen verwendet.

    Parameter:
    true_labels (List[int]): Die Liste der tatsächlichen Labels.
    predicted_labels (List[int]): Die Liste der vorhergesagten Labels.

    Rückgabewert:
    Tuple[List[int], List[int]]: Zwei Listen (true_labels und predicted_labels),
    die auf die gleiche Länge aufgefüllt wurden. 
    """
    max_length = max(len(true_labels), len(predicted_labels))
    
    if len(true_labels) < max_length:
        # Werte aus pred_labels nutzen, um true_labels zu füllen
        for i in range(len(true_labels), max_length):
            true_labels.append(-predicted_labels[i % len(predicted_labels)])
        
    if len(predicted_labels) < max_length:
        # Werte aus true_labels nutzen, um predicted_labels zu füllen
        for i in range(len(predicted_labels), max_length):
            predicted_labels.append(-true_labels[i % len(true_labels)])
        
    return true_labels, predicted_labels

def convert_labels(labels):
    """
    Konvertiert die gegebenen Labels in binäre Werte.

    Parameter:
    labels (List[int]): Eine Liste von Labels, die konvertiert werden sollen.

    Rückgabewert:
    List[int]: Eine Liste von binären Werten
    """
    return [1 if label == 1 else 0 for label in labels]

def evaluate(eval_file, eval_label, additional_infos = ""):
    """
    Diese Funktion bewertet die Vorhersagegenauigkeit eines Modells, indem sie die 
    vorhergesagten Labels aus einer Textdatei mit den tatsächlichen Labels aus einer 
    CSV-Datei vergleicht. Die Funktion berechnet den F1-Score für jede Zeile und 
    führt eine Analyse durch, wie viele Zeilen korrekt oder falsch vorhergesagt wurden. 
    Die Ergebnisse werden in einer Textdatei gespeichert, die den Namen des Modells und 
    der Evaluierungsdaten enthält.

    Parameter:
    - eval_file: Pfad zur Textdatei, die die zu bewertenden Logs enthält.
    - eval_label: Pfad zur CSV-Datei, die die tatsächlichen Labels enthält.
    - additional_infos: (optional) Zusätzliche Informationen, die in der Ausgabedatei 
      gespeichert werden sollen.

    Rückgabewert:
    - Keine Rückgabe. Die Ergebnisse werden in einer Textdatei gespeichert und auf der 
      Konsole ausgegeben.
    """
    file_name = os.path.basename(eval_file)
    model_name = os.path.basename(os.path.normpath(model_load.model_path))
    
    with open(eval_file, 'r') as txt_f, open(eval_label, 'r') as csv_f:
        txt_zeilen = txt_f.readlines()
        csv_reader = csv.reader(csv_f)
        csv_zeilen = [list(map(int, row)) for row in csv_reader]
        
        if len(txt_zeilen) != len(csv_zeilen):
            raise ValueError("Die Dateien haben nicht die gleiche Anzahl an Zeilen")
        
        gleiche_zeilen = 0
        unterschiedliche_zeilen = 0
        miss_eval_content = []
        miss_eval_label = []
        correct_label = []
        f1_scores = []
        for i in range(len(txt_zeilen)):
            print(i)
            pred_labels = model_load.predict_and_display(txt_zeilen[i].strip(), False)
            true_labels = csv_zeilen[i]
            true_labels_padded, pred_labels_padded = pad_labels(true_labels, pred_labels)
            f1 = f1_score(convert_labels(true_labels_padded), convert_labels(pred_labels_padded), zero_division= 1)
            f1_scores.append(f1)
            if pred_labels == true_labels:
                gleiche_zeilen += 1
            else:
                miss_eval_content.append(txt_zeilen[i].strip())
                unterschiedliche_zeilen += 1
                miss_eval_label.append(str(pred_labels))
                correct_label.append(str(true_labels))
                print(f"Zeile {i+1}: Unterschied gefunden")
                print(f"TXT: {pred_labels}")
                print(f"CSV: {true_labels}")
        
        gesamt_zeilen = len(txt_zeilen)
        gleiche_prozent = (gleiche_zeilen / gesamt_zeilen) * 100
        unterschiedliche_prozent = (unterschiedliche_zeilen / gesamt_zeilen) * 100
        
        with open(f"Datensätze/Vorbereitete Daten - Beispiel/03_Evaluationen/{file_name}_{model_name}_evaluation.txt", 'w') as infos:
            infos.write(f"Model: {model_name}\n Evaluierungsdaten: {file_name}\n Sonstige Infos: {additional_infos}\n")
            average_f1 = sum(f1_scores) / len(f1_scores)
            infos.write(f"Durchschnittlicher F1-Score: {average_f1}\n")
            print(f"Durchschnittlicher F1-Score: {average_f1}")
            infos.write(f"Ergebnisse: \n    -Gleiche Zeilen: {gleiche_zeilen} ({gleiche_prozent:.2f}%)\n    -Unterschiedliche Zeilen: {unterschiedliche_zeilen} ({unterschiedliche_prozent:.2f}%)\nFalsch interpretierte Zeilen:\n")

            for line, e_label, label in zip(miss_eval_content, miss_eval_label, correct_label):
                infos.write("Logeintrag: " + line + "\n")
                infos.write("Modell-Label: " + e_label + "\n")
                infos.write("Echtes Label: " + label + "\n")
                infos.write("\n")
        print(f"Gleiche Zeilen: {gleiche_zeilen} ({gleiche_prozent:.2f}%)")
        print(f"Unterschiedliche Zeilen: {unterschiedliche_zeilen} ({unterschiedliche_prozent:.2f}%)")
    


# for input_file, log_file, label_file, output_content, output_label in zip(path.content_file_path_list, path.twok_log_path_list, path.label_list_path_list , path.twok_evaluate_content_list, path.twok_evaluate_label_list):
#     select_random_lines(input_file, log_file, label_file, output_content, output_label)
add_infos = input("Sonstige Infos hinzufügen:")
evaluate(path.twok_content_path_list[4], path.twok_label_path_list[4], add_infos)

# for content, label in zip(path.twok_content_path_list, path.twok_label_path_list):
#     evaluate(content, label, add_infos)

#evaluate(path.twok_content_path_list[4], path.twok_label_path_list[4], add_infos)