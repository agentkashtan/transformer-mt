import os

from tokenizers import ByteLevelBPETokenizer


def get_tokenizer(language, training_data=None):
    VOCAB_SIZE = 30522
    if language not in ["en", "fr"]:
        raise ValueError("Invalid language")

    TOKENIZER_PATH = {
            "en": "tokenizer_en",
            "fr": "tokenizer_fr"
    }
    special_tokens=['<s>', '<pad>', '</s>', '<unk>']
    tokenizer = None
    if os.path.exists(f'{TOKENIZER_PATH[language]}/vocab.json') and os.path.exists(f'{TOKENIZER_PATH[language]}/merges.txt'):
        tokenizer = ByteLevelBPETokenizer(
            f'{TOKENIZER_PATH[language]}/vocab.json',
            f'{TOKENIZER_PATH[language]}/merges.txt'
        )
        tokenizer.add_special_tokens(special_tokens)
    else:
        if training_data is None:
            raise ValueError("Tokenizer does not exist and training data was not provided")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(training_data, vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=special_tokens)
        os.makedirs(TOKENIZER_PATH[language], exist_ok=True)
        tokenizer.save_model(TOKENIZER_PATH[language])
    
    return tokenizer

