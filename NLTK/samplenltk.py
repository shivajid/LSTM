#This python file will read word document and tokenize it.
#I am also using pickle to pickle some data to the table so that it does not need to parse every time
#


import nltk
import tensorflow as tf
import csv
import itertools
import  pickle

vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print "Reading CSV file..."
#with open('/Users/shivajidutta/tensorflow_examples/rnn-tutorial-rnnlm/data/reddit-comments-2015-08.csv', 'rb') as f:

def get_sentences_from_file(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
    return sentences

sentences =  get_sentences_from_file("/Users/shivajidutta/tensorflow_examples/rnn-tutorial-rnnlm/data/reddit-comments-2015-08.csv")

print "dumping sentences"
pickle.dump( sentences, open( "/Users/shivajidutta/tensorflow_examples/rnn-tutorial-rnnlm/data/sentences.p", "wb" ) )

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# This nltk.FreqDist  changes words to frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print (word_freq.most_common(vocabulary_size-1))

print "Found %d unique words tokens." % len(word_freq.items())

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])


