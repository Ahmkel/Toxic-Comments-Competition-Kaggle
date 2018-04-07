import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataGenerator:
    def __init__(self, config):
        vocab_size = config.vocab_size
        all_classes = config.all_classes
        max_length = config.max_length
        padding = config.padding
        train_file = '../data/train.csv'
        test_file = '../data/test.csv'

        data_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        list_sentences_train = data_df["comment_text"].fillna("NA").values
        list_sentences_test = test_df["comment_text"].fillna("NA").values

        if all_classes:
            self.y = data_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
        else:
            y = data_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
            self.y = y.clip(0, 1)

        tokenizer = Tokenizer(num_words=vocab_size, lower=True, oov_token='UNK')
        tokenizer.fit_on_texts(list_sentences_train)
        
        sequences = tokenizer.texts_to_sequences(list_sentences_train)
        self.word_index = tokenizer.word_index
        self.data = pad_sequences(sequences, maxlen=max_length, padding=padding)

        sequences = tokenizer.texts_to_sequences(list_sentences_test)
        self.test = pad_sequences(sequences, maxlen=max_length, padding=padding)

    def get_train_data(self):
        return self.data, self.y

    def get_test_data(self):
        return self.test

    def get_word_index(self):
        return self.word_index