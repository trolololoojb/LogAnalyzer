import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# Beispiel-Datensätze
sentences = [
    ["Der", "Benutzer", "XY", "hat", "sich", "an", "PC5", "angemeldet"],
    ["Die", "IP-Adresse", "192.168.0.1", "wurde", "verwendet"],
    ["Server", "ABC", "hat", "einen", "Fehler", "gemeldet"]
]

labels = [
    ["statisch", "statisch", "variabel", "statisch", "statisch", "statisch", "variabel", "statisch"],
    ["statisch", "statisch", "variabel", "statisch", "statisch"],
    ["statisch", "variabel", "statisch", "statisch", "statisch", "statisch"]
]

# Wörter und Labels zu Indizes kodieren
word2idx = {w: i + 2 for i, w in enumerate(set([w for s in sentences for w in s]))}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

label2idx = {"statisch": 0, "variabel": 1}

# Sätze und Labels in Indizes umwandeln
X = [[word2idx[w] if w in word2idx else word2idx["UNK"] for w in s] for s in sentences]
y = [[label2idx[l] for l in lbl] for lbl in labels]

# Padding der Sequenzen
max_len = max([len(s) for s in sentences])
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Labels one-hot kodieren
y = [to_categorical(i, num_classes=len(label2idx)) for i in y]

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(np.array(y_train))
# Modell erstellen
model = Sequential()
model.add(Embedding(input_dim=len(word2idx), output_dim=50, input_length=max_len))
model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(label2idx), activation="softmax")))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Modell trainieren
history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.2, verbose=1)

# Modell evaluieren
score = model.evaluate(X_test, np.array(y_test), verbose=1)
print(f"Test Accuracy: {score[1]*100:.2f}%")

# Vorhersagen
predictions = model.predict(X_test)

# Beispielvorhersage anzeigen
i = 0  # Index der zu zeigenden Vorhersage
print("Satz:")
print([list(word2idx.keys())[list(word2idx.values()).index(w)] for w in X_test[i] if w != 0])
print("Vorhersagte Labels:")
print([list(label2idx.keys())[np.argmax(label)] for label in predictions[i] if np.max(label) > 0])
print("Echte Labels:")
print([list(label2idx.keys())[np.argmax(label)] for label in y_test[i] if np.max(label) > 0])
