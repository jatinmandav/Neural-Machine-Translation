# Neural-Machine-Translation

### Dataset

German - English translation dataset is downloaded from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/).

First a [CSV file](dataset/english-spanish-dataset.csv) generated using `preprocess.py` for easy management and access.

During training, dataset is divided randomly in 7:3 ratio.

### Embeddings

The embedding layer of the network is initialized first with pre-trained embedding trained using Word2Vec (skipgram model) on [english-spanish-dataset.csv](dataset/english-spanish-dataset.csv) to improve the overall performance of the network.

You can find trained embedding under "RELEASES" section.

## LSTM-Attention Model

### Architecture

Here a simple model with Luong's Attention is used to achieve the task of language translation although more complex models can be built to improve the model's performance. You can read more about basics and backgrounds of NMTs at [tensorflow/nmt](https://github.com/tensorflow/nmt#basic).

I attempt to implement [Luong's attention](https://arxiv.org/pdf/1508.04025.pdf) due to its simplicity in understanding as well as implementation.

This diagram explains the basic architecture of the model [Source: [background-on-the-attention-mechanism](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism)]:
<p align="center"> <img src="results/attention_mechanism.jpg"/> </p>


### Code

#### Preprocessing

Dataset from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/) is in the format of text file with language1-lagnuage2 seperated by `\t` (tab). For better management, I have opt to create a `CSV` file. `preprocess.py` takes input the `.txt` (`spa.txt`) and cleans the file by converting non-printable characters to printable format and generates the `CSV` file.

```
usage: preprocess.py [-h] --dataset DATASET [--language_1 LANGUAGE_1]
                     --language_2 LANGUAGE_2 [--save_file SAVE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Path to .txt file downloaded from
                        http://www.manythings.org/anki/
  --language_1 LANGUAGE_1, -l1 LANGUAGE_1
                        Language-1 name | Default: english
  --language_2 LANGUAGE_2, -l2 LANGUAGE_2
                        Language-2 name
  --save_file SAVE_FILE, -s SAVE_FILE
                        Name of CSV file to be generated | Default:
                        dataset.csv

```

#### Generating Embeddings

You can generate word2vec embeddings using `generate_embeddings.py` on your own dataset.
```
usage: generate_embeddings.py [-h] --csv CSV --language LANGUAGE [--sg]
                              [-emb_size EMB_SIZE] [--save_file SAVE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --csv CSV             Path to CSV which is to be read
  --language LANGUAGE, -l LANGUAGE
                        Language whose embedding is to be generated from CSV
                        file. (It should be the column name in CSV)
  --sg                  Whether to use SkipGram or CBoW model | Default: CBoW
  -emb_size EMB_SIZE, -s EMB_SIZE
                        Size of embedding to generate | Defualt: 256
  --save_file SAVE_FILE
                        Name of embedding file to save | Default:
                        [skipgram/cbow]-[language]-[embedding_size].model

```

You are now ready to train the model.

#### Training the Model

To train your model, run `train.py` from terminal by supplying the following arguments.

```
usage: train.py [-h] --dataset DATASET [--language_1 LANGUAGE_1] --language_2
                LANGUAGE_2 [--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM]
                [--lang1_embedding LANG1_EMBEDDING]
                [--lang2_embedding LANG2_EMBEDDING]
                [--train_val_split TRAIN_VAL_SPLIT] [--epochs EPOCHS]
                [--log_dir LOG_DIR] [--inference]
                [--pretrained_path PRETRAINED_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Path to Training Dataset
  --language_1 LANGUAGE_1, -l1 LANGUAGE_1
                        Language-1 name | Default: english
  --language_2 LANGUAGE_2, -l2 LANGUAGE_2
                        Language-2 name
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        What should be the Batch Size? | Default: 16
  --latent_dim LATENT_DIM, -dim LATENT_DIM
                        What should be the latent dimensions? | Default: 256
  --lang1_embedding LANG1_EMBEDDING, -le1 LANG1_EMBEDDING
                        Path to Language-1 Embeddings Word2Vec model
  --lang2_embedding LANG2_EMBEDDING, -le2 LANG2_EMBEDDING
                        Path to language-2 Embeddings Word2Vec model
  --train_val_split TRAIN_VAL_SPLIT, -tvs TRAIN_VAL_SPLIT
                        Train-vs-Validation Split ratio | Default: 0.3
  --epochs EPOCHS, -es EPOCHS
                        Number of epochs to train on | Default: 30
  --log_dir LOG_DIR, -l LOG_DIR
                        Where to save tensorboard graphs and trained weights?
                        | Default: nmt_logs
  --inference           Whether to run inference or simply train the network
  --pretrained_path PRETRAINED_PATH
                        Path to Pre-trained Weights
```

You can also run `tensorboard` to monitor the train-vs-val loss. The weights and tensorboard logs will be saved in `log_dir`.
