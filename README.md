# Neural-Machine-Translation

### Dataset

German - English translation dataset is downloaded from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/).

First a [CSV file](english-german-dataset.csv) generated using `preprocess.py` for easy management and access.

Dataset is divided in following size randomly:
 - Training Size: 138876 sentence pairs (90% of Training dataset)
 - Validation Size: 15430 sentence pairs (10% of Training dataset)
 - Testing Size: 38576 sentence pairs (20% of total dataset)

### Embeddings

The embedding layer of the network is initialized first with pre-trained embedding trained using Word2Vec (skipgram model) on [english-german-dataset.csv](dataset/english-german-dataset.csv) to improve the overall performance of the network.

Vocabulary Size:
 - English: 15711
 - German: 33161

You can find trained embedding under "RELEASES" section.

## LSTM-Attention Model

### Architecture

Here a simple model with Luong's Attention is used to achieve the task of language translation although more complex models can be built to improve the model's performance. You can read more about basics and backgrounds of NMTs at [tensorflow/nmt](https://github.com/tensorflow/nmt#basic).

I attempt to implement [Luong's attention](https://arxiv.org/pdf/1508.04025.pdf) due to its simplicity in understanding as well as implementation.

This diagram explains the basic architecture of the model [Source: [background-on-the-attention-mechanism](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism)]:
<p align="center"> <img src="results/attention_mechanism.jpg"/> </p>


### Code

#### Preprocessing

Dataset from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/) is in the format of text file with language1-lagnuage2 seperated by `\t` (tab). For better management, I have opt to create a `CSV` file. `preprocess.py` takes input the `.txt` (`deu.txt`) and cleans the file by converting non-printable characters to printable format and generates the `CSV` file.

```
usage: preprocess.py [-h] --dataset DATASET [--save_file SAVE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Path to .txt file downloaded from
                        http://www.manythings.org/anki/
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
