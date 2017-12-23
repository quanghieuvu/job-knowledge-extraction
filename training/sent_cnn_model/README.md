# How to Run this

This is the version of the DSSM model with word embeddings input layer and multiple convolutional layers on top of that.

### I Get the dataset

Dowload the question/topic dataset [here](https://raw.githubusercontent.com/scottyih/STAGG/master/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv)
and save it in the `data` folder.

Download the pretrain word embedding [here](https://worksheets.codalab.org/rest/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/contents/blob/)
and save it in the folder `data/word_representations` with the name `glove.6B.100d.txt`

Run the data_helpers.py script to verify that everything is ok.

``` sh
python data_helpers.py
```


### II Train

To train the network run:

``` sh
python train_cnn.py
```

You can monitor your training on tensorboard by running:

``` sh
tensorboard --logdir runs/{YOURUNNUMBER}
```