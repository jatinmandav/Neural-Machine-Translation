# Neural-Machine-Translation

### Dataset

German - English translation dataset is downloaded from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/).

First a [CSV file](english-german-dataset.csv) generated using `preprocess.py` for easy management and access.

Dataset is divided in following random size:
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

This diagram explain's the basic architecture of the model:
<p align="center"> <img src="results/attention_mechanism.jpg"/> </p>
