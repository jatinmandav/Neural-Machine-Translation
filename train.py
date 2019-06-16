from model import NMTModelDef
from ReadData import ReadData

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import nltk
from tqdm import tqdm
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import seaborn

seaborn.set(font=['AppleMyungjo'], font_scale=1)

import argparse

def prep_string_to_input(X):
    batch_size = 1
    for j in range(0, 1, batch_size):
        encoder_input_data = np.zeros((batch_size, reader.max_length_english),dtype='float32')
        #decoder_input_data = np.zeros((batch_size, reader.max_length_german),dtype='float32')
        #decoder_target_data = np.zeros((batch_size, reader.max_length_german, reader.num_decoder_tokens),dtype='float32')
        for i, input_text in enumerate(X[j:j+batch_size]):
            for t, word in enumerate(input_text.split()):
                encoder_input_data[i, t] = reader.input_token_index[word] # encoder input seq
    return encoder_input_data


# Decoding output of network to translated sequence and getting Attention layer output
def attent_and_generate(input_seq):
    decoded_sentence = []

    [encoder_output, h, c] = encoder_model.predict(input_seq)
    states_value = [h, c]

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = reader.target_token_index['START_']

    stop_condition = False
    attention_density = []
    index = []

    while not stop_condition:
        output_tokens, h, c, attention = attention_model.predict([target_seq, encoder_output] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reader.reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        if ('_END' in sampled_char) or len(decoded_sentence) > 50:
            stop_condition = True

        states_value = [h, c]
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        attention_density.append((sampled_char, attention[0][0]))

    return np.array(attention_density), ' '.join(decoded_sentence)

# Visualizing Attention Plot
def visualize(text, encoder_input):
    attention_weights, decoded_sent = attent_and_generate(encoder_input)

    plt.clf()
    plt.figure(figsize=(10,10))

    mats = []
    dec_inputs = []
    for dec_ind, attn in attention_weights:
        mats.append(attn[:len(text[0].split(' '))].reshape(-1))
        dec_inputs.append(dec_ind)

    attention_mat = np.transpose(np.array(mats))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(attention_mat)
    ax.set_xticks(np.arange(attention_mat.shape[1]))
    ax.set_yticks(np.arange(attention_mat.shape[0]))

    ax.set_xticklabels([inp for inp in dec_inputs])
    ax.set_yticklabels([w for w in str(text[0]).split(' ')])

    ax.tick_params(labelsize=15)
    ax.tick_params(axis='x', labelrotation=90)

    plt.show()
    return decoded_sent

# Custom TensorBoard to visualize Training-vs-Validation training
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
parser.add_argument('--language_1', '-l1', help='Language-1 name | Default: english', default='english')
parser.add_argument('--language_2', '-l2', help='Language-2 name', required=True)
parser.add_argument('--batch_size', '-bs', help='What should be the Batch Size? | Default: 16', default=16, type=int)
parser.add_argument('--latent_dim', '-dim', help='What should be the latent dimensions? | Default: 256', default=256, type=int)
parser.add_argument('--lang1_embedding', '-le1', help='Path to Language-1 Embeddings Word2Vec model', default='embeddings/skipgram-english-256.model')
parser.add_argument('--lang2_embedding', '-le2', help='Path to language-2 Embeddings Word2Vec model', default='embeddings/skipgram-spanish-256.model')
parser.add_argument('--train_val_split', '-tvs', help='Train-vs-Validation Split ratio | Default: 0.3', default=0.3, type=float)
parser.add_argument('--epochs', '-es', help='Number of epochs to train on | Default: 30', default=30, type=int)
parser.add_argument('--log_dir', '-l', help='Where to save tensorboard graphs and trained weights? | Default: nmt_logs', default='nmt_logs')
parser.add_argument('--inference', action="store_true", help='Whether to run inference or simply train the network')
parser.add_argument('--pretrained_path', help='Path to Pre-trained Weights')

args = parser.parse_args()

assert args.dataset.endswith('csv'), "Dataset File needs to be in CSV format"
assert 0. <= args.train_val_split < 1., "Train-vs-Validation Split need to be between [0, 1)"

latent_dim = args.latent_dim

# Reading and Preparing Training/Validation Dataset
reader = ReadData(args.dataset, args.train_val_split, args.language_1, args.language_2)
(X_train, y_train), (X_val, y_val) = reader.prep_data()
train_samples = len(X_train)
val_samples = len(X_val)
num_encoder_tokens = reader.num_encoder_tokens
num_decoder_tokens = reader.num_decoder_tokens

# Loading Embedding Matrix
lang1_embedding = Word2Vec.load(args.lang1_embedding)
lang1_tok = Tokenizer()
lang1_tok.fit_on_texts(reader.language_1_text)

encoder_embedding_matrix = np.zeros((num_encoder_tokens, latent_dim))
for word, i in lang1_tok.word_index.items():
    try:
        embedding_vector = lang1_embedding[word]
        if embedding_vector is not None:
            encoder_embedding_matrix[i] = embedding_vector
    except Exception as e:
        pass

lang2_embedding = Word2Vec.load(args.lang2_embedding)
lang2_tok = Tokenizer()
lang2_tok.fit_on_texts(reader.language_2_text)

decoder_embedding_matrix = np.zeros((num_decoder_tokens, latent_dim))
for word, i in lang2_tok.word_index.items():
    try:
        embedding_vector = lang2_embedding[word]
        if embedding_vector is not None:
            decoder_embedding_matrix[i] = embedding_vector
    except Exception as e:
        pass

# Defining Model
nmt = NMTModelDef(latent_dim, num_encoder_tokens, num_decoder_tokens,
                encoder_embedding_matrix, decoder_embedding_matrix, args.pretrained_path)

if not args.inference:
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

else:
    model, encoder_model, decoder_model = nmt.build(inference=True)
    assert(model.layers[7] == model.get_layer('attention'))

    attention_layer = decoder_model.get_layer('attention') # or model.layers[7]
    attention_model = Model(inputs=decoder_model.inputs, outputs=decoder_model.outputs + [attention_layer.output])
    sent = [X_train[5]]
    encoder_input = prep_string_to_input(sent)
    decoded_sent = visualize(sent, encoder_input)
