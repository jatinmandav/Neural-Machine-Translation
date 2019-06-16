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
    def __init__(self, path, train_val_split, language_1, language_2):
        self.path = path
        self.train_val_split = train_val_split
        self.dataset = pd.read_csv(self.path)
        self.dataset = self.dataset[[language_1, language_2]]
        self.language_1 = language_1
        self.language_2 = language_2
        assert language_2 in self.dataset.columns and language_1 in self.dataset.columns, "Invalid format of {} file.".format(self.path)

    def prep_data(self):
        self.clean_data()
        self.get_info()

        print(self.dataset.head(10))

        self.language_1_text = list(self.dataset[self.language_1])
        self.language_2_text = list(self.dataset[self.language_2])

        x = []
        y = []
        for i in range(len(self.language_1_text)):
            x.append(str(self.language_1_text[i]))
            y.append(str(self.language_2_text[i]))

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.train_val_split)

        return (X_train, y_train), (X_test, y_test)

    def clean_data(self):
        # Uncasing all sentences pairs
        self.dataset[self.language_1] = self.dataset[self.language_1].apply(lambda x: x.lower())
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x: x.lower())

        # Removing Punctuation
        exclude = set(string.punctuation)
        self.dataset[self.language_1] = self.dataset[self.language_1].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

        # Removing unwanted trailing characters
        self.dataset[self.language_1] = self.dataset[self.language_1].apply(lambda x: x.strip())
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x: x.strip())

        self.dataset[self.language_1] = self.dataset[self.language_1].apply(lambda x: re.sub(" +", " ", x))
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x:  re.sub(" +", " ", x))

        self.dataset[self.language_1] = self.dataset[self.language_1].apply(lambda x: re.sub("\?\?", "", x))
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x:  re.sub("\?\?", "", x))

        # Adding "START_" and "_END" tokens
        self.dataset[self.language_2] = self.dataset[self.language_2].apply(lambda x: "START_ " + x + " _END")

    def get_info(self):
        vocab_language_1 = set()
        for sent in self.dataset[self.language_1]:
            for word in sent.split():
                if word not in vocab_language_1:
                    vocab_language_1.add(word)

        vocab_language_2 = set()
        for sent in self.dataset[self.language_2]:
            for word in sent.split():
                if word not in vocab_language_2:
                    vocab_language_2.add(word)

        length_list = []
        for l in self.dataset[self.language_1]:
            length_list.append(len(l.split(' ')))
        self.max_length_language_1 = np.max(length_list)

        length_list = []
        for l in self.dataset[self.language_2]:
            length_list.append(len(l.split(' ')))
        self.max_length_language_2 = np.max(length_list)

        input_words = sorted(list(vocab_language_1))
        target_words = sorted(list(vocab_language_2))

        self.num_encoder_tokens = len(vocab_language_1)
        self.num_decoder_tokens = len(vocab_language_2) + 1

        self.input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
        self.target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

        self.reverse_input_char_index = dict((i, word) for word, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, word) for word, i in self.target_token_index.items())

        self.dataset = shuffle(self.dataset)

    def generate_batch(self, X, y, batch_size):
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, self.max_length_language_1),dtype='float32')
                decoder_input_data = np.zeros((batch_size, self.max_length_language_2),dtype='float32')
                decoder_target_data = np.zeros((batch_size, self.max_length_language_2, self.num_decoder_tokens),dtype='float32')
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
                            decoder_target_data[i, t - 1, self.target_token_index[word]] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)
