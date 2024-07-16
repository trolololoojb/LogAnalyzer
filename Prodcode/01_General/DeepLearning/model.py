import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

# Modellparameter
embedding_dim = 64
gru_units = 128
num_classes = 2  # Klassifikation als 0 oder 1

# Eingabesequenz der Token-IDs
inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

# Word Embedding Schicht
x = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim)(inputs)

# Bidirektionale GRU-Schicht
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))(x)

# Dense Schicht zur Klassifikation
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Zusammenbau des Modells
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Kompilieren des Modells
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modellzusammenfassung
model.summary()

# Callbacks definieren
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, mode='min')


# Modelltraining mit Callbacks
model.fit(X_padded, Y_onehot, epochs=100, batch_size=16, verbose=1,
          validation_split=0.2,  # Angenommen, Sie möchten eine Validierung durchführen
          callbacks=[checkpoint, early_stopping, reduce_lr])

# Aktuelles Datum und Uhrzeit für den Modellnamen
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f'{current_time}_logAnalyzer.keras'