from gensim.models import Word2Vec
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--csv', help='Path to CSV which is to be read', required=True)
parser.add_argument('--language', '-l', help='Language whose embedding is to be generated from CSV file. (It should be the column name in CSV)',
                    required=True)
parser.add_argument('--sg', action='store_true', help='Whether to use SkipGram or CBoW model | Default: CBoW')
parser.add_argument('-emb_size', '-s', help='Size of embedding to generate | Defualt: 256', default=256, type=int)
parser.add_argument('--save_file', help='Name of embedding file to save | Default: [skipgram/cbow]-[language]-[embedding_size].model')

args = parser.parse_args()

data = pd.read_csv(args.csv)
data = data[args.language]

words = [sent.split(' ') for sent in list(data)]
model = Word2Vec(words, min_count=1, workers=4, sg=args.sg, size=args.emb_size)

print('Vocab Size: {}'.format(len(list(model.wv.vocab))))

if args.sg:
    emb_type = 'skipgram'
else:
    emb_type = 'cbow'

model.save('{}-{}-{}.model'.format(emb_type, args.language, args.emb_size))
