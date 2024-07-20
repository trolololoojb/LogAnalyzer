
from tokenizers import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer.json")
model = load_model(r'Datensätze/Vorbereitete Daten - Beispiel/Models/Model.keras')
# Evaluation des Modells
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')
max_length=102
# Funktion zur Ausgabe der Vorhersagen
def predict_and_display(log):
    encode = tokenizer.encode(log)
    sequence_padded = pad_sequences([encode.ids], maxlen=max_length, padding='post')
    print(sequence_padded)
    prediction = model.predict(sequence_padded)[0]

    words = log.split()
    for word, pred in zip(words, prediction):
        label = 'nicht statisch' if pred > 0.5 else 'statisch'
        print(f'Wort: {word}, Vorhersage: {label}')

# Beispielvorhersage
new_log = "Received block blk_-9999999 of size 99 from /99.99.999.99"
print("\nVorhersagen für neuen Logeintrag:")
predict_and_display(new_log)