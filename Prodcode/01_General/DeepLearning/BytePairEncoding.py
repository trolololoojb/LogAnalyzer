import re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, processors
import path

def BPE_labels(subwords, labels):
    """
    Ordnet den BPE-Subwörtern die passenden Labels zu.

    Parameter:
    subwords (list of str): Liste der BPE-Subwörter. 
    labels (list of int): Liste der ursprünglichen Labels.

    Rückgabewert:
    new_labels (list of int): Labels, die den BPE-Subwörtern zugeordnet sind.
    """
    new_labels = []
    counter = 0
    for subword in subwords:
        if  re.search("§", subword): 
            new_labels.append(labels[counter])
            counter += 1
        else:
            new_labels.append(labels[counter])
    return new_labels

def generateTokenizer_BPE(text, files:bool = False):
    """
    Erstellt einen BPE (Byte Pair Encoding) Tokenizer und speichert ihn als JSON-Datei.

    Parameters:
    text (list of str or str): Ein einzelner Text oder eine Liste von Texten, 
                               die zum Trainieren des Tokenizers verwendet werden.
                               Wenn 'files' True ist, sollten dies Dateipfade sein.
    files (bool): Ein Flag, das angibt, ob 'text' eine Liste von Dateipfaden ist. 
                  Wenn False, wird 'text' als eine Liste von Textstrings behandelt.

    Returns:
    tokenizer (Tokenizer): Ein trainierter BPE-Tokenizer.
    """
    # Initialisierung des Tokenizers mit BPE-Modell
    tokenizer = Tokenizer(models.BPE())

    # Festlegen der Normalisierungsregeln (optional)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase()
    ])

    # Festlegen der Pre-Tokenisierung, um Leerzeichen als Token zu erfassen
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])

    # Training des Tokenizers mit einem Trainer, der die Leerzeichen einschließt
    trainer = trainers.BpeTrainer(
        special_tokens=["<pad>", "<unk>", " "],
        show_progress=True
    )
    if files:
        tokenizer.train(files=text, trainer=trainer)
    else:
        tokenizer.train_from_iterator(iterator=text, trainer=trainer)

    # Post-Processing, um sicherzustellen, dass Leerzeichen als Token behandelt werden
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer



