# How to Run this

This is the DSSM version based on Word Embeddings and Convolutional Layers.

### I Prepare the Dataset

Dowload the Kaggle job post dataset [here](https://www.kaggle.com/madhab/jobposts/downloads/online-job-postings.zip) and save it into the `data` folder.

Download the pretrain word embedding [here](https://worksheets.codalab.org/rest/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/contents/blob/) and save it into the folder `data/word_representations` with the name `glove.6B.100d.txt`

Run the sent_cnn_data_helpers, that will execute the `job_posting_data_preprocessing` function. This will create your formated dataset into the data folder as `job_post_preprocessed.tsv`

``` sh
cd {ROOT_DIR}/preprocessing
python sent_cnn_data_helpers.py
```

### II Train

To train the network run:

``` sh
cd {ROOT_DIR}/training/sent_cnn_model
python train_cnn.py
```

You can monitor your training on tensorboard by running:

``` sh
tensorboard --logdir runs/{YOURUNNUMBER}
```


> You can also run this on the question/topics dataset which is giving really good results (>92% accuracy on the validation set). You can download it [here](https://raw.githubusercontent.com/scottyih/STAGG/master/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv)