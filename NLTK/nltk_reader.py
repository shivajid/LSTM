#This is sample file to pickle and unpickle a file.

import pickle
import nltk


sentences = pickle.load(open("/Users/shivajidutta/tensorflow_examples/rnn-tutorial-rnnlm/data/sentences.p", "rb"))
print (sentences[2])
