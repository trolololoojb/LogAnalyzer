def extract_words(sentences, marker):
    """
    Funktion zum Extrahieren von großgeschriebenen Wörtern
    """
    uppercase_words = []
    for sentence in sentences:
        words = sentence.split()
        if marker:
            #Es wird str.istitle() verwendet, da Kürzungen wie wm sp etc. nicht zum Ergebnis beitragen. Kann bei Bedarf durch
            # word[0].isupper() or word[0].isdigit() ersetzt werden.  
            uppercase_words.extend([word for word in words if word.istitle()])
        else:
            if len(words) > 1:
                words = ['unklar']
            uppercase_words.extend(words)
    return uppercase_words

import re
def word_shortener(list):
    """
    Entfernt Sonderzeichen, Streicht Füllwörter und schreibt alle Wörter klein
    """
    clean_text = ""
    exclude =["die", "das", "ein", "auch", "der", "an", "am", "bei", "eine", "im", "in", "aus", "kein"]
    for word in list:
        word = word.lower()
        word = re.sub(r'[^A-Za-z0-9äöüÄÖÜß\s]', '', word)
        word = re.sub(r'[^A-Za-zäöüÄÖÜß\s]', '0', word) 
        if word not in exclude:
            clean_text += word + " "
    return clean_text

import csv



def transform(csv_file_path, marker):
    """
    Wandelt eine CSV Datei in eine 2 dimensionale Liste um.
    Wörter werden in klein geschrieben umgewandelt, Sonderzeichen werden ersetzt und gewisse Füllwörter gestrichen.
    Marker entscheidet ob nur groß geschriebene Wörter in der CSV auch verarbeitet werden. Außerdem werden dann EInträge mit zwei Wörtern durch unklar ersetzt.\n
    Args:
        csv_file_path (String): Pfad zu der CSV Datei
        marker (Boolean): True - Nur groß geschriebene Wörter, False - Alle Wörter
    """
    data_list = []
    # Einlesen der CSV-Datei. Pandas wurde nicht genutzt,
    # da es mit der CSV Datei nicht klar gekommen ist
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        next(csvreader)  # Überspringen der ersten Zeile (Header)
        for row in csvreader:
            data_list.append(word_shortener(extract_words(row, marker)))
    return data_list


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def tokenizer(tokenizer ,list:list, padding:bool, fit:bool = False):
    """
    Tokenisiert die übergebene Liste.\n
    Args:
    tokenizer (Tokenizer): Ein Keras Tokenizer Objekt.
    texts (List[str]): Eine Liste von Texten, die tokenisiert werden sollen.
    padding (bool): Gibt an, ob die Sequenzen gepaddet werden sollen. true = padding
    fit (bool): Gibt an ob der tokenizer noch mal auf den text angepasst werden soll. Standard False. false = ja.
    Tokenizer wird übergeben um eine spätere Rückumwandlung zu ermöglichen.
    """
    tokenizer = tokenizer
    if not fit:
        tokenizer.fit_on_texts(list)
    index_bib = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(list)
    if padding:
        sequences = pad_sequences(sequences, padding = 'post')
    return sequences, index_bib
