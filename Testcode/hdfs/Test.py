# import numpy as np
# import tensorflow as tf
# import csv


# # Pfad zur CSV-Datei
# file_path = r"C:\Users\j-u-b\OneDrive\Studium\Semester 6\Bachelorarbeit\Code\LogAnalyzer\Testcode\hdfs\token_list_test.csv"


# np.set_printoptions(threshold=np.inf)
# X_data = []

# # Angenommen, die Datei 'output.csv' existiert und enthält Daten im CSV-Format
# with open(file_path, 'r', newline='', encoding='utf-8') as file:
#     reader = csv.reader(file, delimiter= ";")
#     for row in reader:
#         # Konvertiere jede Zahl von String zu Integer
#         #converted_row = [int(num) for num in row]
#         X_data.append(row)


# print(X_data)

# X_data_padded = tf.keras.preprocessing.sequence.pad_sequences(X_data, padding='post')
# print(X_data_padded)


#-------------------------------------------------------------------------------


# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Beispielhafte Logdaten generieren
# logs = [
#     "INFO User logged in",
#     "ERROR Failed to connect to database",
#     "WARN Low disk space",
#     "INFO User logged out",
#     "ERROR Unexpected error occurred"
# ]

# # Labels für die Logtypen
# labels = ["INFO", "ERROR", "WARN"]

# # Daten vorbereiten
# X = np.array(logs)
# y = np.array([0, 1, 2, 0, 1])  # Indizes der Labels

# # Label-Encoding der Zielvariablen
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Tokenisierung und Padding der Sequenzen
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# X_tokenized = tokenizer.texts_to_sequences(X)
# X_padded = pad_sequences(X_tokenized, padding='post')

# # Train-Test-Split
# X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2)

# # Modell definieren
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_padded.shape[1]))
# model.add(LSTM(100))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(len(labels), activation='softmax'))

# # Modell kompilieren
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Modell trainieren
# model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# # Modell evaluieren
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # Beispielvorhersage
# new_logs = ["INFO User login attempt", "ERROR Disk failure"]
# new_logs_tokenized = tokenizer.texts_to_sequences(new_logs)
# new_logs_padded = pad_sequences(new_logs_tokenized, padding='post', maxlen=X_padded.shape[1])

# predictions = model.predict(new_logs_padded)
# predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# print(predicted_labels)





#----------------------------------------------------------------------------------------------------------------

list = []

list1 = [1,1,1,1,1,1]

list2 = [2,2,2,2,2,2,2]


list += list1
list += list2

print(list)








