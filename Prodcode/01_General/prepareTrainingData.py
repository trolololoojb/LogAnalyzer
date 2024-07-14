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

