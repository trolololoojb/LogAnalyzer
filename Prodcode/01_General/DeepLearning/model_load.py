
import os
from tokenizers import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = r'Datensätze/Vorbereitete Daten - Beispiel/01_Models/20240725-002519**** (gekappter bgl)/tokenizedModel.keras'
tokenizer_path = r"Datensätze/Vorbereitete Daten - Beispiel/01_Models/20240725-002519**** (gekappter bgl)/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
model = load_model(model_path)
directory_path = os.path.dirname(model_path)

# Eingabelänge extrahieren
max_length_file_path = os.path.join(directory_path, 'max_length.txt')

# max_length aus der Datei lesen
try:
    with open(max_length_file_path, 'r') as file:
        max_length = int(file.read().strip())
    print(f'Die maximale Eingabelänge ist: {max_length}')
except:
    max_length = input("Keine max-length Datei! Zum Fortfahren max_length angeben:")

def predict_and_display(log):


    sequence = [tokenizer.encode(log).ids]
    sequence_padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    print(sequence_padded[0])
    prediction = model.predict(sequence_padded)[0]

    words = tokenizer.encode(log).tokens
    result = []
    current_label = 'variabel' if prediction[0] > 0 else 'statisch'
    current_words = [words[0]]
    current_values = [prediction[0]]

    for word, pred in zip(words[1:], prediction[1:]):
        label = 'variabel' if pred > 0 else 'statisch'
        if label == current_label:
            current_words.append(word)
            current_values.append(pred)
        else:
            result.append((current_words, current_label, current_values))
            current_words = [word]
            current_values = [pred]
            current_label = label

    # Letzte Gruppe hinzufügen
    if current_words:
        result.append((current_words, current_label, current_values))

    # Ausgabe und Schreiben in Datei
    for group, label, values in result:
        text = ''.join(group)
        values_str = ', '.join(map(str, values))
        print(f'Wörter: {text}, Vorhersage: {label}, Werte: {values_str}')

new_log = input("Log eingeben: ")
print("\nVorhersagen für neuen Logeintrag:")
predict_and_display(new_log)