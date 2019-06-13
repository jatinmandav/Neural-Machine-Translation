import pandas as pd
import numpy as np
import re
import os
import nltk
from tqdm import tqdm
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class ReadData:
    def __init__(self, path, train_val_split):
        self.path = path
        self.train_val_split = train_val_split
        self.dataset = pd.read_csv(self.path)
        assert 'english' in self.dataset.columns and 'german' in self.dataset.columns, "Invalid format of {} file.".format(self.path)

    def prep_data(self):
        self.clean_data()
        self.get_info()

        self.english = list(self.dataset.english)
        self.german = list(self.dataset.german)

        x = []
        y = []
        for i in range(len(self.english)):
            x.append(str(self.english[i]))
            y.append(str(self.german[i]))

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.train_val_split)

        return (X_train, y_train), (X_test, y_test)

    def clean_data(self):
        # Uncasing all sentences pairs
        self.dataset.english = self.dataset.english.apply(lambda x: x.lower())
        self.dataset.german = self.dataset.german.apply(lambda x: x.lower())

        # Removing Punctuation
        exclude = set(string.punctuation)
        self.dataset.english = self.dataset.english.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        self.dataset.german = self.dataset.german.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

        # Removing unwanted trailing characters
        self.dataset.english = self.dataset.english.apply(lambda x: x.strip())
        self.dataset.german = self.dataset.german.apply(lambda x: x.strip())

        self.dataset.english = self.dataset.english.apply(lambda x: re.sub(" +", " ", x))
        self.dataset.german = self.dataset.german.apply(lambda x:  re.sub(" +", " ", x))

        self.dataset.english = self.dataset.english.apply(lambda x: re.sub("\?\?", "", x))
        self.dataset.german = self.dataset.german.apply(lambda x:  re.sub("\?\?", "", x))

        # Adding "START_" and "_END" tokens
        self.dataset.english = self.dataset.german.apply(lambda x: "START_" + x + "_END")

    def get_info(self):
        vocab_english = set()
        for sent in self.dataset.english:
            for word in sent.split():
                if word not in vocab_english:
                    vocab_english.add(word)

        vocab_german = set()
        for sent in self.dataset.german:
            for word in sent.split():
                if word not in vocab_german:
                    vocab_german.add(word)

        length_list = []
        for l in self.dataset.english:
            length_list.append(len(l.split(' ')))
        self.max_length_english = np.max(length_list)

        length_list = []
        for l in self.dataset.german:
            length_list.append(len(l.split(' ')))
        self.max_length_german = np.max(length_list)

        input_words = sorted(list(vocab_english))
        target_words = sorted(list(vocab_german))

        self.num_encoder_tokens = len(vocab_english)
        self.num_decoder_tokens = len(vocab_german) + 1

        self.input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
        self.target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

        self.reverse_input_char_index = dict((i, word) for word, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, word) for word, i in self.target_token_index.items())

        self.dataset = shuffle(self.dataset)

    def generate_batch(self, X, y, batch_size):
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, self.max_length_english),dtype='float32')
                decoder_input_data = np.zeros((batch_size, self.max_length_german),dtype='float32')
                decoder_target_data = np.zeros((batch_size, self.max_length_german, self.num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        encoder_input_data[i, t] = self.input_token_index[word] # encoder input seq
                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = self.target_token_index[word] # decoder input seq
                        if t>0:
                            # decoder target sequence (one hot encoded)
                            # does not include the START_ token
                            # Offset by one timestep
                            decoder_target_data[i, t - 1, target_token_index[word]] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)
