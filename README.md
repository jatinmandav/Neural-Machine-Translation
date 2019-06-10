# Neural-Machine-Translation

### Dataset

German - English translation dataset is downloaded from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/).

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
