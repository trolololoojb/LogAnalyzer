
from tokenizers import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer.from_file(r"/home/johann/github/LogAnalyzer/Datensätze/Vorbereitete Daten - Beispiel/Tokenizer/tokenizer.json")
model = load_model(r'Datensätze/Vorbereitete Daten - Beispiel/Models/20240722-124954_50/tokenizedModel.keras')

max_length=102
def predict_and_display(log):
    encode = tokenizer.encode(log)
    sequence_padded = pad_sequences([encode.ids], maxlen=max_length, padding='post', value = -1)
    print(encode.tokens)
    print(sequence_padded)
    prediction = model.predict(sequence_padded)[0]
    print()

    words = encode.tokens
    for word, pred in zip(words, prediction):
        label = 'nicht statisch' if pred > 0.5 else 'statisch'
        print(f'Wort: {word}, Vorhersage: {label}, Wert: {pred}')

new_log = "9 ddr errors(s) detected and corrected on rank 9, symbol 9, bit 9"
print("\nVorhersagen für neuen Logeintrag:")
predict_and_display(new_log)