import numpy as np
import re
import os

def get_evaluation_examples_for_sent2vec(training_file_name, n_neg_sample=5):
    neg_dict = {}
    wqn_lst = []
    x_u = []
    x_r = []
    y = []
    with open(training_file_name) as tf:
        for lines in tf:
            # data -> [f1, u_str, r_str, WQ_NUM]
            data = lines.split('\t')
            f1 = float(data[0])
            u_str = clean_str(data[1].strip())
            r_str = data[2].strip()
            wqn = data[3].strip()
            
            if f1 >= 0.5:
                wqn_lst.append(wqn)
                x_u.append(u_str)
                x_r.append([r_str])
            else:
                if neg_dict.has_key(wqn):
                    if len(neg_dict[wqn]) < n_neg_sample:
                        neg_dict[wqn].append(r_str)
                else:
                    neg_dict[wqn] = [r_str]
        for i in range(len(x_u)):
            if not neg_dict.has_key(wqn_lst[i]):
                neg_dict[wqn_lst[i]] = neg_dict[wqn_lst[0]]
            if len(neg_dict[wqn_lst[i]]) < n_neg_sample:
                neg_dict[wqn_lst[i]] += neg_dict[wqn_lst[0]][:n_neg_sample - len(neg_dict[wqn_lst[i]])]
            if len(neg_dict[wqn_lst[i]]) != n_neg_sample:
                print neg_dict[wqn_lst[i]]
            x_r[i] = x_r[i] + neg_dict[wqn_lst[i]]
            # add a little randomness 
            y.append(np.random.randint(len(x_r[i])))
            tmp = x_r[i][0]
            x_r[i][0] = x_r[i][y[i]]
            x_r[i][y[i]] = tmp
    
    with open("sent2vec.eval", 'w') as f:
        for i, xu in enumerate(x_u):
            for xr in x_r[i]:
                line = ""
                line = xu + '\t' + re.sub(r"\.", " ", xr) + '\n'
                f.write(line)
                
    return x_u, x_r, y 

def get_training_examples(training_file_name, n_neg=5000):
    """
    Load training data file, and split the data into words and labels.
    Return utterances, relation words, 
    labels and the largest length of training sentences.
    """
    positive_utterances = []
    positive_relations = []
    negative_utterances = []
    negative_relations = []
    max_len = 0
    with open(training_file_name) as tf:
        for lines in tf:
            # data -> [f1, u_str, r_str, WQ_NUM]
            data = lines.split('\t')
            f1 = float(data[0])
            u_str = clean_str(data[1].strip()).split(" ")
            r_str = clean_str(data[2].strip()).split(" ")
            max_len = max(len(u_str), len(r_str), max_len)
            if f1 >= 0.5:
                positive_utterances.append(u_str)
                positive_relations.append(r_str)
            elif f1 == 0:
                negative_utterances.append(u_str)
                negative_relations.append(r_str)
        samples = np.random.choice(len(negative_utternaces), n_neg, replace=False)
        negative_utternaces = [negative_utternaces[i] for i in samples]
        negative_relations = [negative.relations[i] for i in samples]
        x_u = positive_utterances + negative_utterances
        x_r = positive_relations + negative_relations
        positive_labels = np.ones(len(positive_pairs))
        negative_labels = np.ones(len(negative_pairs))
        y = np.concatenate((positive_labels, negative_labels))
    
    return (x_u, x_r, y, max_len)

def get_training_examples_for_softmax(training_file_name, n_neg_sample=5):
    """
    Load training data file, and split the data into words and labels.
    Return utterances, relation words, 
    labels and the largest length of training sentences.
    The output of this funtion is formatted for softmax regression.
    """
    neg_dict = {}
    max_len = 0
    wqn_lst = []
    x_u = [] # Question
    x_r = [] # [[[topics2], [topicspos2].. etc]]
    y = []   # [id_of_the_good_topic_sequence]
    with open(training_file_name) as tf:
        for lines in tf:
            # data -> [f1, u_str, r_str, WQ_NUM]
            data = lines.split('\t')
            f1 = float(data[0])
            u_str = clean_str(data[1].strip()).split(" ")
            r_str = clean_str(data[2].strip()).split(" ")
            wqn = data[3].strip()
            max_len = max(len(u_str), len(r_str), max_len)
            
            if f1 >= 0.5:
                wqn_lst.append(wqn)
                x_u.append(u_str)
                x_r.append([r_str])
            else:
                if neg_dict.has_key(wqn):
                    if len(neg_dict[wqn]) < n_neg_sample:
                        neg_dict[wqn].append(r_str)
                else:
                    neg_dict[wqn] = [r_str]
        for i in range(len(x_u)):
            if not neg_dict.has_key(wqn_lst[i]):
                neg_dict[wqn_lst[i]] = neg_dict[wqn_lst[0]]
            if len(neg_dict[wqn_lst[i]]) < n_neg_sample:
                # TODO: Add some randomness to this selection process
                neg_dict[wqn_lst[i]] += neg_dict[wqn_lst[0]][:n_neg_sample - len(neg_dict[wqn_lst[i]])]
            if len(neg_dict[wqn_lst[i]]) != n_neg_sample:
                print neg_dict[wqn_lst[i]]
            x_r[i] = x_r[i] + neg_dict[wqn_lst[i]]
            # add a little randomness 
            y.append(np.random.randint(len(x_r[i])))
            tmp = x_r[i][0]
            x_r[i][0] = x_r[i][y[i]]
            x_r[i][y[i]] = tmp
    
    return (x_u, x_r, y, max_len)
        

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(sentence, length, padding_word="pad"):
    """
    Pad a sentence to a given length.
    """
    num_padding = length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    return new_sentence
    

def get_pretrained_wordvec_from_file(wf, dim):
    """
    Get the pre-trained word representations from local file(s).
    
    @param wf: path to the file
    @type wf: str
    
    @param dim: the dimension of embedding matrix.
    @type dim: tuple
    
    @return (tokens, U): tokens row dict and words embedding matrix. 
    @rtype : tuple of (dict, nparray)
    """
    tokens = {}
    U = np.zeros(dim)
    
    with open(wf) as f:
        for inc, lines in enumerate(f):
            tokens[lines.split()[0]] = inc
            U[inc] = np.array(map(float, lines.split()[1:]))

            if inc == dim[0]:
                break
    return tokens, U


import pandas as pd
import random

def clean_custom(to_process):
    to_process = to_process.lower()
    to_process = to_process.replace("\n", " ")  # take off \n at the end of line
    to_process = re.sub(r'[^\w\s]', ' ', to_process)  # replace punctuation with ' '
    to_process = "".join([c for c in to_process if (c.isalnum() or c == " ")])  # take off all non alnum char
    to_process = " ".join(to_process.split())  # Remove multiple spaces and join with * "a    b"= "a*b"
    return to_process

def job_posting_data_preprocessing(input_file_path):
    data = pd.read_csv(input_file_path)
    data = data[["JobDescription", "Title"]]
    print data.count()
    data = data.dropna(axis=0, how='any')
    print data.count()
    np_data = data.as_matrix()

    with open(os.path.join(os.path.dirname(input_file_path), "job_post_preprocessed.tsv"), "w") as f:
        for index in range(np_data.shape[0]):
            # Cutting job post after 50 words (Simple case for now)
            job_description = " ".join(np_data[index][0].split()[:50])
            title = np_data[index][1]
            title = clean_custom(title).replace(" ", ".")
            job_description = clean_custom(job_description)
            random_sampling = []
            while len(random_sampling) != 10 or index in random_sampling or len(set(random_sampling)) != 10:
                random_sampling = [random.randrange(0, np_data.shape[0]) for x in range(10)]
            random_sampling = [clean_custom(np_data[x][1]).replace(" ", ".") for x in random_sampling]
            random_sampling = zip([title] + random_sampling, [1] + [0]* 10)
            random.shuffle(random_sampling)
            [f.write(str(x[1]) + "\t" + job_description + "\t" + x[0] + "\t" + "WebQTrn-{}".format(index) + "\n") for x in random_sampling]

            if index % 1000 == 0:
                print "Processing {} on {} examples".format(index, np_data.shape[0])

if __name__ == "__main__":
    # tokens, U = get_pretrained_wordvec_from_file("./data/word_representations/glove.6B.100d.txt", (400000, 100))
    job_posting_data_preprocessing("../data/data job posts.csv")