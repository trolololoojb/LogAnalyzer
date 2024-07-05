import dataSetTransform as dst
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from datetime import datetime
#CSV Dateien:
freitext_file = r"C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\Freitext.csv"
ursache_file = r"C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\Ursache.csv"


ursache_list = dst.transform(ursache_file, False)
freitext_list = dst.transform(freitext_file, True)


#Tokensisierung der beiden Listen und Padding des Freitextes
tokenizer_x = Tokenizer()
X_padded, index_bib_x = dst.tokenizer(tokenizer_x, freitext_list, True)
tokenizer_y = Tokenizer()
sequences_ursache, index_bib_y = dst.tokenizer(tokenizer_y, ursache_list, False)
np.set_printoptions(threshold=np.inf)



num_words = len(index_bib_x) + 1
vector_dim = 80
len_pads = 143
output_dim = len(index_bib_y) + 1

Y_onehot = to_categorical(sequences_ursache, num_classes=output_dim)


# Tokenizer in JSON-Format konvertieren und speichern
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
tokenizer_json_x = tokenizer_x.to_json()
with open(f'{current_time}_predictiveMaintenanceTokenizer_x.json', 'w') as f:
    f.write(tokenizer_json_x)

tokenizer_json_y = tokenizer_y.to_json()
with open(f'{current_time}_predictiveMaintenanceTokenizer_y.json', 'w') as f:
    f.write(tokenizer_json_y)


model = Sequential()
model.add(Embedding(input_dim=num_words,
                    output_dim=vector_dim,
                    mask_zero=True))
model.add(SimpleRNN(units=vector_dim))
model.add(Dense(units=output_dim, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# Callbacks definieren
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, mode='min')

# Aktuelles Datum und Uhrzeit für den Modellnamen
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Modelltraining mit Callbacks
model.fit(X_padded, Y_onehot, epochs=100, batch_size=16, verbose=1,
          validation_split=0.2,  # Angenommen, Sie möchten eine Validierung durchführen
          callbacks=[checkpoint, early_stopping, reduce_lr])

# Modell speichern
model_name = f'{current_time}_predictiveMaintenance.keras'
model.save(model_name)
