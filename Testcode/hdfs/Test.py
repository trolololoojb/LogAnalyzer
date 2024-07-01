from tensorflow.keras.preprocessing.text import Tokenizer

reviews = ['Die Sonne scheint',
           'Es regnet heute wieder',
           'Bewölkt',
           'Regen!',
           'Es regnet nicht']

tok = Tokenizer()
tok.fit_on_texts(reviews)
X_seq = tok.texts_to_sequences(reviews)

tok.index_word, X_seq

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_pad = pad_sequences(X_seq, maxlen=3)
X_pad

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

num_words = len(tok.word_index) + 1
vector_dim = 2
len_pads = 3

model = Sequential()
model.add(Embedding(input_dim=num_words,
                    output_dim=vector_dim,
                    input_length=len_pads,
                    mask_zero=True))
model.add(SimpleRNN(units=vector_dim))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
model.summary()

import numpy as np
y = np.array([0, 1, 0, 1, 0])
model.fit(X_pad, y, epochs=100, batch_size=1, verbose=False)
word_vec = model.get_layer(index=0).get_weights()
print(word_vec)


# Vorhersage für den neuen Satz
new_review = ["Die Sonne scheint"]
new_seq = tok.texts_to_sequences(new_review)
new_pad = pad_sequences(new_seq, maxlen=3)

# Vorhersage
prediction = model.predict(new_pad)
print("Vorhersage für 'Es regnet morgen wieder':", prediction)