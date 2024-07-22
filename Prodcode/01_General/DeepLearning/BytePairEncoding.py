import keras_nlp
from tensorflow.keras.preprocessing.text import Tokenizer

# Beispiel-Text
texts = [
    "Logeintrag eins",
    "Logeintrag zwei",
    "Logeintrag drei",
    "Logeintrag vier"
]

# Initialisieren des BytePairTokenizers
# Hinweis: In einem realen Szenario müssen Sie den Tokenizer mit Ihrem eigenen Vokabular trainieren.
# Hier ist ein einfaches Beispiel, wie Sie den Tokenizer einrichten könnten.
# Die tatsächliche Implementierung kann variieren und erfordert normalerweise ein Trainingsset.

# Ein einfaches Beispiel zum Laden eines vortrainierten Tokenizers
tokenizer = keras_nlp.tokenizers.BytePairTokenizer(
    vocabulary_size=1000,
    sequence_length=100,
)

# Trainieren des Tokenizers (dieser Schritt wird normalerweise mit einem großen Textkorpus durchgeführt)
# Hier verwenden wir Beispieltexte nur zur Veranschaulichung.
tokenizer.train_on_texts(texts)

# Text tokenisieren
encoded_texts = tokenizer(texts)

# Ausgabe des tokenisierten Textes
print("Tokenisierte Texte:")
for i, encoded_text in enumerate(encoded_texts):
    print(f"Text {i+1}: {encoded_text.numpy()}")
