# -*- coding: utf8 -*-
import numpy as np
import re
import itertools
from collections import Counter


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


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def loadTrainOrTest_data(data_path, label_path, vocabulary_word2index):
    x_text = list(open(data_path,"r").readlines())
    x_temp = list([a.strip().split(" ") for a in x_text])
    x = [[a.strip() for a in b]  for b in x_temp]
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = vocabulary_word2index.get(x[i][j],0)
    x_train = np.array(x).tolist()
    y_text = list(open(label_path,"r").readlines())
    y_train = np.array([np.int32(a) for a in y_text])
    return x_train, y_train

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    lengths = [len(s) for s in sequences]
    #compyter the maxlen of sentense 
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    print("maxlen=",maxlen)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        #如果是pre，则截取后面maxlen的词，如果是post，截取前maxlen的词
        if truncating == 'post':
            trunc = s[:maxlen]
        elif truncating == 'pre':
            trunc = s[-maxlen:]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x