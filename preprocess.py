import nltk
import string
from unicodedata import normalize
import numpy as np
import re
import pickle
from tqdm import tqdm
import pandas as pd

class PreprocessData:
    def __init__(self, path):
        self.path = path
        self.load_data()

    def load_data(self):
        with open(self.path, mode='rt', encoding='utf-8') as f:
            self.text = f.read().strip().split('\n')

        self.text_pairs = []
        for line in self.text:
            self.text_pairs.append(line.split('\t'))

    def clean_text(self):
        cleaned = []
        re_print = re.compile('[^%s]' % re.escape(string.printable))

        table = str.maketrans('', '', string.punctuation)
        for pair in self.text_pairs:
            clean_pair = []
            for line in pair:
                line = normalize('NFD', line).encode('ascii', 'ignore')
                line = line.decode('UTF-8')
                line = line.split(' ')

                line = [word.lower() for word in line]
                line = [word.translate(table) for word in line]
                line = [re_print.sub('', word) for word in line]
                line = [word for word in line if word.isalpha()]

                clean_pair.append(' '.join(line))

            cleaned.append(clean_pair)

        return cleaned

preprocess = PreprocessData('dataset/deu.txt')
cleaned_text = preprocess.clean_text()
english = []
german = []

for pair in tqdm(cleaned_text):
    english.append(pair[0])
    german.append(pair[1])

df = pd.DataFrame([english, german])
df = df.transpose()
df.columns = ['english', 'german']
df.to_csv('dataset/english-german-dataset.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

test_split = 0.2
test_data = df.head(int(len(df)*test_split)).reset_index(drop=True)
train_data = df.tail(int(len(df)*(1-test_split))).reset_index(drop=True)

print('Train Size: {}, Test Size: {}'.format(len(train_data), len(test_data)))
test_data.to_csv('dataset/english-german-test.csv')
test_data.to_csv('dataset/english-german-train.csv')
