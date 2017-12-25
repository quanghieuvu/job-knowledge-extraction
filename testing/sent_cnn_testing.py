import tensorflow as tf
import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__), "../../")))
import training.sent_cnn_model.config as config
import preprocessing.sent_cnn_data_helpers as dh

def format_data_for_testing(job_post, titles):
    """
    Load training examples and pretrained word embeddings from disk.
    Return training inputs, labels and pretrianed embeddings.
    """
    # Load raw data
    titles = [dh.clean_custom(x).split() for x in titles]
    job_post = dh.clean_custom(job_post).split()
    job_post = job_post[:50]
    max_len = max(len(job_post), max([len(x) for x in titles]))

    # Pad sentences
    pad = lambda x: dh.pad_sentences(x, max_len)
    pad_lst = lambda x: map(pad, x)
    x_u = [dh.pad_sentences(job_post, max_len)]
    x_r = [[dh.pad_sentences(x, max_len) for x in titles]]

    # Load tokens and pretrained embeddings
    we_file = config.config["word_embeddings_file"]
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

CHECKPOINT_DIR = "/home/pierre/Documents/Upwork/Code/job-knowledge-extraction/training/sent_cnn_model/runs/1514198447/checkpoints/"

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(CHECKPOINT_DIR, "sent_cnn_3_8-100.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print "Model has been restored"
    job_post = "We are currently looking to grow our group with senior and junior engineers that have experience in developing and deploying NLP, Reinforcement Learning, Anomaly Detection, Customer Service or similar applications at scale. The ideal candidate will have experience working as a technical lead or junior team members in a start-up, industrial, government, or academic lab setting on artificial intelligence and machine learning projects."
    titles = ["Chief Financial Officer", "BCC Specialist", "Software Developer" ,"Assistant to Managing Director", "Chief Accountant/ Finance Assistant", "Software Engineer, Machine Learning"]

    x_u_i, x_r_i, max_len, U = format_data_for_testing(job_post, titles)
    input_x_u = graph.get_tensor_by_name("input_x_u")
    input_x_r = graph.get_tensor_by_name("input_x_r")
    dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob")
    predictions = graph.get_tensor_by_name("cosine_layer/predictions:0")
    y = sess.run(predictions, feed_dict={input_x_u: x_u_i, input_x_r: x_r_i, dropout_keep_prob: 1})
    print y