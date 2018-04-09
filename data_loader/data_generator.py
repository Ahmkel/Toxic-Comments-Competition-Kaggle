import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class DataGenerator:
    def __init__(self, config):
        all_classes = config.all_classes
        max_sequence_length = config.max_sequence_length
        padding = config.padding
        train_file = 'datasets/train.csv'
        test_file = 'datasets/test.csv'

        data_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        list_sentences_train = data_df["comment_text"].fillna("NA").values
        list_sentences_test = test_df["comment_text"].fillna("NA").values

        if all_classes:
            self.y = np.array(data_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])
        else:
            self.y = np.array(data_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1).clip(0, 1))

        if config.get('vocab_size') is None:
            config.vocab_size = None

        tokenizer = Tokenizer(num_words=config.vocab_size, lower=True, oov_token='UNK')
        tokenizer.fit_on_texts(list_sentences_train)

        self.word_index = tokenizer.word_index
        if config.get('vocab_size') is None:
            config.vocab_size = len(self.word_index)

        sequences = tokenizer.texts_to_sequences(list_sentences_train)
        self.data = pad_sequences(sequences, maxlen=max_sequence_length, padding=padding)

        sequences = tokenizer.texts_to_sequences(list_sentences_test)
        self.test = pad_sequences(sequences, maxlen=max_sequence_length, padding=padding)

    def get_train_data(self):
        return self.data, self.y

    def get_test_data(self):
        return self.test

    def get_word_index(self):
        return self.word_index
