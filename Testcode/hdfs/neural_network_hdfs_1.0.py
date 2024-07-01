import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
#from tensorflow.keras.layers import (Dense, Conv1D,MaxPool1D, Flatten)
from sklearn.model_selection import train_test_split
import ast

content_list_file = r"Datensätze\HDFS\content_list.txt"
label_list_file = r"Datensätze\HDFS\label_list.txt"

# Importieren der Contentlist
with open(content_list_file, 'r') as file:
    data = file.read()
    content_list_big = ast.literal_eval(data)

# Umwandeln jeder Zeichenkette in der Liste in Kleinbuchstaben
content_list = [wort.lower() for wort in content_list_big]

# Importieren der Labellist
with open(label_list_file, 'r') as file:
    data = file.read()
    label_list = ast.literal_eval(data)

# Erstellen eines Indexwörterbuchs
vocab = {word: idx for idx, word in enumerate(set(" ".join(content_list).split()), start=1)}

# Ersetzen der Wörter durch die Indizes
tokenized_logs = [[vocab[word] for word in log.split()] for log in content_list]

# Bestimmen der maximalen Länge der Wörter und Padding
maxlen = max([len(log) for log in tokenized_logs])
max_len =maxlen
tokenized_logs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_logs, maxlen=maxlen, padding='post')
labels = tf.keras.preprocessing.sequence.pad_sequences(label_list, maxlen=maxlen, padding='post')#

with open('tokenized_log_padding.txt', 'w') as file:
    file.write(str(tokenized_logs))

with open('labels_padding.txt', 'w') as file:
    file.write(str(labels))

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_val, y_train, y_val = train_test_split(tokenized_logs, labels, test_size=0.2, random_state=42)

print("Train labels shape before:", y_train.shape)
print("Validation labels shape before:", y_val.shape)

import numpy as np

# Erweitern der Labels um eine zusätzliche Dimension
y_train = np.expand_dims(y_train, -1)
y_val = np.expand_dims(y_val, -1)


# Modellarchitektur
num_words = len(vocab) + 1
vector_dim = 95

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=vector_dim, mask_zero=True))
model.add(Bidirectional(GRU(units=vector_dim, return_sequences=True)))
model.add(Dense(units=1, activation='sigmoid'))

model.build(input_shape=(None, maxlen))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='model_firsttry_hdfs.keras',
    monitor='val_loss',
    save_best_only=True
)


history = model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_data= (X_val, y_val), callbacks=[stopping, checkpoint])


# # Gewichte anzeigen
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)


def predict_log(model, vocab, log, max_len):
    # Text vorbereiten
    tokenized_log = [vocab.get(word, 0) for word in log.lower().split()]  # Konvertiere jeden Wort zu einem Index, nutze 0 für unbekannte Wörter
    padded = tf.keras.preprocessing.sequence.pad_sequences([tokenized_log], maxlen=max_len, padding='post')  # Führe Padding durch
    
    # Vorhersage machen
    prediction = model.predict(padded)
    return (prediction > 0.5).astype(int)  # Schwellenwert festlegen


# Beispiel für eine Vorhersage
test_log = "An PC 42 hat sich User ZZ angemeldet"
print(predict_log(model, tokenizer, test_log, max_len))

