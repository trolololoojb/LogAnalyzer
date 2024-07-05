from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import dataSetTransform as dst

# Modell laden
model = load_model(r'C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\Modelle\2024_07_04_1809_45E_57P\20240704-180946_predictiveMaintenance.keras')

# Tokenizer aus JSON wiederherstellen
with open(r'C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\Modelle\2024_07_04_1809_45E_57P\20240704-180946_predictiveMaintenanceTokenizer_x.json') as f:
    tokenizer_json_x = f.read()
tokenizer_x = tokenizer_from_json(tokenizer_json_x)

with open(r'C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\Modelle\2024_07_04_1809_45E_57P\20240704-180946_predictiveMaintenanceTokenizer_y.json') as f:
    tokenizer_json_y = f.read()
tokenizer_y = tokenizer_from_json(tokenizer_json_y)

testsequence = r"C:\Users\johann.bartels\OneDrive - IT-P DE Intern\Desktop\predictive-maintenance\test.csv"
testsequence = dst.transform(testsequence, True)
test_padded, test_index = dst.tokenizer(tokenizer_x, testsequence, True, True)
prediction = model.predict(test_padded)
print(prediction)
predicted_word = tokenizer_y.index_word[np.argmax(prediction)]
print("Vorhergesagtes Wort:", predicted_word)
