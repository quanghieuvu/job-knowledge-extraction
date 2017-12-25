import tensorflow as tf
import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__), "../../")))
import training.sent_cnn_model.config as config
import preprocessing.sent_cnn_data_helpers as dh
import numpy as np

def format_data_for_testing(job_post, titles, max_len=50):
    """
    Prepare the test set for the network.
    """
    # Preprocess data
    titles = [dh.clean_custom(x).split() for x in titles]
    job_post = dh.clean_custom(job_post).split()
    job_post = job_post[:50]

    # Pad sentences
    x_u = [dh.pad_sentences(job_post, max_len)]
    x_r = [[dh.pad_sentences(x, max_len) for x in titles]]

    # Load tokens and pretrained embeddings
    we_file = "/".join(config.config["word_embeddings_file"].split("/")[1:])
    voc_size = config.config["vocabulary_size"]
    embedding_size = config.config["embedding_size"]
    tokens, U = dh.get_pretrained_wordvec_from_file(we_file, (voc_size, embedding_size))

    # Represent sentences as list(nparray) of ints
    dctize = lambda word: tokens[word] if tokens.has_key(word) else tokens["pad"]
    dctizes = lambda words: map(dctize, words)
    dctizess = lambda wordss: map(dctizes, wordss)
    x_u_i = np.array(map(dctizes, x_u))
    x_r_i = np.array(map(dctizess, x_r))
    
    return (x_u_i, x_r_i, max_len, U)

CHECKPOINT_DIR = "/home/pierre/Documents/Upwork/Code/job-knowledge-extraction/training/sent_cnn_model/runs/1514224634/checkpoints/"

# TODO Implement inference for only one vector.
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(CHECKPOINT_DIR, "sent_cnn_3_8-200.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print "Model has been restored"
    job_post = "We are currently looking for a senior to develop our application. The main technologies used will be react and little bit of angular. Also we have some cordova for the mobile app"
    titles = ["BCC Specialist", "Chief Financial Officer" ,"Assistant to Managing Director", "Software Developer", "Chief Accountant/ Finance Assistant", "Software Engineer, Machine Learning"]

    graph = tf.get_default_graph()

    max_len = graph.get_tensor_by_name("sequence_length:0")
    max_len = sess.run(max_len)

    x_u_i, x_r_i, max_len, U = format_data_for_testing(job_post, titles, max_len=max_len)

    input_x_u = graph.get_tensor_by_name("input_x_u:0")
    input_x_r = graph.get_tensor_by_name("input_x_r:0")
    dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
    predictions = graph.get_tensor_by_name("cosine_layer/predictions:0")
    y = sess.run(predictions, feed_dict={input_x_u: x_u_i, input_x_r: x_r_i, dropout_keep_prob: 1})
    print y