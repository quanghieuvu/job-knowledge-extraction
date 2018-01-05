config = {
    "data_file": "../../data/job_post_preprocessed.tsv",
    "word_embeddings_file": "../../data/word_representations/glove.6B.100d.txt",
    "vocabulary_size": 400000,
    "embedding_size": 100,
    "num_classes": 6,
    "filter_sizes": [3, 4, 5],
    "num_filters": 8,
    "dropout_keep_prob": 0.75,
    "embeddings_trainable": True,
    "total_iter": 100000,
    "batch_size": 256,
    "val_size": 1000,
    "l2_reg_lambda": 0.1,
    "checkpoint_step": 200,
    "max_len": 50,
    "double_net": True
}