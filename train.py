from model import NMTModelDef
from ReadData import ReadData

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import nltk
from tqdm import tqdm
import os

import argparse

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='Path to Training Dataset', required=True)
parser.add_argument('--batch_size', '-bs', help='What should be the Batch Size? | Default: 16', default=16, type=int)
parser.add_argument('--latent_dim', '-dim', help='What should be the latent dimensions? | Default: 256', default=256, type=int)
parser.add_argument('--english_embedding', '-e', help='Path to English Embeddings Word2Vec model', default='embeddings/skipgram-english-256.model')
parser.add_argument('--german_embedding', '-g', help='Path to German Embeddings Word2Vec model', default='embeddings/skipgram-german-256.model')
parser.add_argument('--train_val_split', '-tvs', help='Train-vs-Validation Split ratio | Default: 0.2', default=0.2, type=float)
parser.add_argument('--epochs', '-es', help='Number of epochs to train on | Default: 30', default=30, type=int)
parser.add_argument('--log_dir', '-l', help='Where to save tensorboard graphs and trained weights? | Default: eng_ger_nmt_logs', default='eng_ger_nmt_logs')

args = parser.parse_args()

assert args.dataset.endswith('csv'), "Dataset File needs to be in CSV format"
assert 0. <= args.train_val_split < 1., "Train-vs-Validation Split need to be between [0, 1)"

latent_dim = args.latent_dim

# Reading and Preparing Training/Validation Dataset
reader = ReadData(args.dataset, args.train_val_split)
(X_train, y_train), (X_val, y_val) = reader.prep_data()
train_samples = len(X_train)
val_samples = len(X_val)
num_encoder_tokens = reader.num_encoder_tokens
num_decoder_tokens = reader.num_decoder_tokens

# Loading Embedding Matrix
english_embedding = Word2Vec.load(args.english_embedding)
eng_tok = Tokenizer()
eng_tok.fit_on_texts(reader.english)

encoder_embedding_matrix = np.zeros((num_encoder_tokens, latent_dim))
for word, i in eng_tok.word_index.items():
    try:
        embedding_vector = english_embedding[word]
        if embedding_vector is not None:
            encoder_embedding_matrix[i] = embedding_vector
    except Exception as e:
        pass

german_embedding = Word2Vec.load(args.german_embedding)
ger_tok = Tokenizer()
ger_tok.fit_on_texts(reader.german)

decoder_embedding_matrix = np.zeros((num_decoder_tokens, latent_dim))
for word, i in ger_tok.word_index.items():
    try:
        embedding_vector = german_embedding[word]
        if embedding_vector is not None:
            decoder_embedding_matrix[i] = embedding_vector
    except Exception as e:
        pass

# Defining Model
nmt = NMTModelDef(latent_dim, num_encoder_tokens, num_decoder_tokens,
                encoder_embedding_matrix, decoder_embedding_matrix)
model = nmt.build(inference=False)

# Keras Callbacks for logging, checkpoints etc.
log_dir = args.log_dir
logging = TrainValTensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'),
                            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

training_generator = reader.generate_batch(X_train, y_train, args.batch_size)
validation_generator = reader.generate_batch(X_val, y_val, args.batch_size)

print('\nRun: `tensorboard --logdir={}`\n'.format(log_dir))

exit()
model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_samples//args.batch_size,
                    epochs=args.epochs,
                    validation_data=validation_generator,
                    validation_steps=val_samples//args.batch_size,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
