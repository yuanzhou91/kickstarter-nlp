# base modeling and model evaluation tools
import tensorflow as tf
import sklearn
from tensorflow import keras
from sklearn import metrics
import sklearn.model_selection
import sys

# utility libraries for math and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import pandas as pd

# utility for determining execution time
# most useful for the training process
from time import time 

# natural language processing and neural network modeling methods
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Lambda
# We will use `one_hot` as implemented by one of the backends
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2



MODEL_NAME = 'LSTM'
ENCODING = 'WORD_EMBEDDINGS'
DROPOUT_RATE = 0.2
STDOOUT_TO_FILE = 'True'

FIG_PREFIX = 'OPTIMIZATION|' + MODEL_NAME + "|" + ENCODING + "|DROPOUT=" + str(DROPOUT_RATE) + "|" 

log_file = open(FIG_PREFIX + "records.log", "w")
if STDOOUT_TO_FILE == 'True':
    sys.stdout = log_file

print('Text Classification with Neural Networks\n')
# information about TensorFlow environment
print('TensorFlow/Keras environment for this work:')
print('Numpy version: ', np.__version__)
print('TensorFlow version: ', tf.__version__)
print('TensorFlow was built with CUDA: ', tf.test.is_built_with_cuda())
print('TensorFlow running with GPU: ', tf.test.is_gpu_available())
print('Keras version: ', tf.keras.__version__)

# TensorFlow eager execution is default in version 2.x.x
# but must be activated if version 1.x.x
if tf.__version__.split('.')[0] == '1':
    tf.enable_eager_execution()

# The numpy version for this test was 1.18.1
# It is possible to have two numpy versions installed on a system.
# This can cause problems. To ensure that a current version is
# installed, it may be necessary to unistall numpy twice 
# prior to installing a new version of numpy:
#     pip uninstall numpy
#     pip uninstall numpy
#     pip install numpy

# set random seed for reproducible results 
# this should work with CPU-only systems
# may not work with GPU-equiped systems
np.random.seed(seed = 9999)
# tf.random.set_seed(9999)

# set up base class for callbacks to monitor training
# and for early stopping during training
tf.keras.callbacks.Callback()

# # read json lines files for the Reduced Reuters Corpus
# # this will provide a list of dictionaries for training and testing
# # dictionary key 'text' is the document text and key 'category' is the label
# fulltraining = [] # list of dictionary structures
# fulltraindocs = [] # list of text strings for documents (get rid of end-of-line characters)
# fulltraincats = [] # list of category labels for documents
# with open ('reuters-corpus-v001/reuters_train.jl', 'r') as f:
#     for line in f:
#         this_line_dictionary = json.loads(line)
#         fulltraining.append(this_line_dictionary)
#         fulltraindocs.append(this_line_dictionary['text'].replace('\n',''))
#         fulltraincats.append(this_line_dictionary['category'])


data = pd.read_csv("df_text_eng.csv") 
# Preview the first 5 lines of the loaded data 
print(data.head())

print(data.columns)
X = data['blurb'].astype(str)
y = data['state'].astype(str)

fulltraindocs, testdocs, fulltraincats, testcats = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

# implementing a tripartite splitting into train, dev, and test
traindocs = fulltraindocs[:130000]
devdocs = fulltraindocs[130000:]
traincats = fulltraincats[:130000]
devcats = fulltraincats[130000:]

# determine the unique number of document types/categories
# and ensure that categories are consistent across train, dev, and test
nclasses = len(set(traincats))
if (set(traincats) != set(devcats)) or (set(traincats) != set(testcats)):
    print('\n\n STOP: Classes inconsistent across train, dev, and test\n\n')

# --------------------------------------------------------------------
# natural language processing model hyperparameters
vocab_size = 5000  # number of unique words to use in tokenization
embedding_dim = 8  # dimension of neural network embedding for a word
max_length = 25    # number of words to be retained in each document
# --------------------------------------------------------------------

# set up tokenizer based on the training documents only
# default filter includes basic punctuation, tabs, and newlines
# filters = !"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
# we add all numbers to this filter 
# default is to convert to lowercase letters
# default is to split on spaces
# oov_token is used for out-of-vocabulary words
tokenizer = Tokenizer(num_words = vocab_size, oov_token = 'OOV',
    filters = '0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n')
tokenizer.fit_on_texts(traindocs)
word_index = tokenizer.word_index 
# word_index is a dictionary of words and their uniquely assigned integers
print('Training vocabulary size with one out-of-vocabulary item: ',
    len(word_index)) 

# execute helper function to create a reverse dictionary
# so if we are given an index value we can retrieve the associated word
reverse_word_index = \
    dict([(value, key) for (key, value) in word_index.items()])

# set up a simple tokenizer for the categories/labels 
# because we have integers replacing category labels, we will use 
# sparse_categorical_crossentropy for the loss function
# when fitting the text classification model
catslist = list(set(traincats))
catsdict = {catslist[i] : i for i in range(0, len(catslist))}
print('\nInteger codes for category labels:')
print(catsdict)

train_labels = []
for word in traincats:
    train_labels.append(catsdict.get(word)) 
train_labels = np.array(train_labels)    

dev_labels = []
for word in devcats:
    dev_labels.append(catsdict.get(word))    
dev_labels = np.array(dev_labels) 

test_labels = []
for word in testcats:
    test_labels.append(catsdict.get(word))
test_labels = np.array(test_labels) 

# create a reverse label function to retrieve the category word 
# from the label/index value
labelsdict = \
    dict([(value, key) for (key, value) in catsdict.items()])
print('\nLabel integers to category words:')
print(labelsdict)    

# define function for preparing input to neural network modeling sequences 
# converting words to sequences of numbers a la tokenization
# for documents with more words than max_length: truncating = 'post'
# for documents with fewer words than max_length: padding = 'post'
def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs) # words to integers
    padded = pad_sequences(encoded, maxlen = max_length, 
        padding = 'post', truncating = 'post', value = 0)
    return padded    

# define model structure with key hyperparameters for multi-class model
def define_model(vocab_size, embedding_dim, max_length, nclasses):
    # define the structure of the model
    model = Sequential()
    model = add_layers(model, vocab_size, embedding_dim, max_length, nclasses)
    # using sparse_categorical_crossentropy because category labels are integers
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    plot_model(model, to_file = FIG_PREFIX + 'fig-model-structure.png', dpi = 128, show_shapes = True, show_layer_names = True)
    return model    


def add_layers(model, vocab_size, embedding_dim, max_length, nclasses):
    model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
    model.add(LSTM(units = 8, activation = 'relu', bias_regularizer=l2(0.01)))
    model.add(Dense(8, activation = 'relu'))
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(nclasses, activation = 'softmax'))
    return model

# encode the documents 
train_sequence = encode_docs(tokenizer, max_length, docs = traindocs)
dev_sequence = encode_docs(tokenizer, max_length, docs = devdocs)
test_sequence = encode_docs(tokenizer, max_length, docs = testdocs)

# set model to run for max_epochs unless early stopping rule is met
# see documentation at https://keras.io/callbacks/
# note that language models may need more epochs than vision models 
# we employ early stopping to shorten the training time
max_epochs = 100
earlystop_callback = \
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',\
    min_delta=0.01, patience=2, verbose=0, mode='auto',\
    baseline=None, restore_best_weights=False)

# fit the model showing progress epoch-by-epoch
# define dev/validation set from within the training set 
# shuffle = False set to obtain reproducible results
# We also use time to compute execution time for training.
# What time is measured depends on the system (Windows versus other).
model = define_model(vocab_size, embedding_dim, max_length, nclasses)

print('\nEpoch-by-Epoch Training Process')
begin_time = time()
history = model.fit(train_sequence, train_labels,
    epochs = max_epochs, shuffle = False,
    validation_data = (dev_sequence, dev_labels), 
    verbose = 2,
    callbacks = [earlystop_callback])
execution_time = time() - begin_time
print('\nTime of execution for training (seconds):', \
	'{:10.3f}'.format(np.round(execution_time, decimals = 3)))

# evaluate fitted model on the full training set
train_loss, train_acc = model.evaluate(train_sequence,train_labels,verbose = 3)
print('\nFull training set accuracy:', \
	'{:6.4f}'.format(np.round(train_acc, decimals = 4)), '\n')

# evaluate the fitted model on the hold-out test set
test_loss, test_acc = model.evaluate(test_sequence,  test_labels, verbose = 3)
print('Hold-out test set accuracy:', \
	'{:6.4f}'.format(np.round(test_acc, decimals = 4)))

# The training process may be evaluated by comparing training and
# dev (validation) set performance. We use "dev" to indicate
# that these data are used in evaluating various hyperparameter
# settings. We do not test alternative hyperparameters here,
# but in other programs there will be much hyperparameter testing.
def plot_history(history):
    print(history.history)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_number = range(1, len(acc) + 1)
    plt.style.use('ggplot') # Grammar of Graphics plots
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_number, acc, 'b', label='Training')
    plt.plot(epoch_number, val_acc, 'r', label='Dev')
    plt.title('Training and Dev Set Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epoch_number, loss, 'b', label='Training')
    plt.plot(epoch_number, val_loss, 'r', label='Dev')
    plt.title('Training and Dev Set Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FIG_PREFIX + 'fig-training-process.pdf', orientation ='landscape')
    plt.close()

# show training process in external visualizations
plot_history(history) 

# examine the predicted values within a precision/recall framework
test_preds = np.argmax(model.predict(test_sequence), axis = 1)

# prepare the classification performance report
category_names = []
for i in range(len(set(traincats))):
    category_names.append(labelsdict.get(i))

print("\nClassification Report for Hold-out Test Set:\n\n%s"
      % (metrics.classification_report(test_labels, test_preds, \
	target_names = category_names)))

# note that the weighted average is most appropriate for classification
# problems that are unbalanced. See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
F1_value = \
    metrics.precision_recall_fscore_support(test_labels, test_preds, average='weighted')[2]
print('Test set F1 (weighted average):', \
	'{:6.4f}'.format(np.round(F1_value, decimals = 4)),'\n')

cm_data = metrics.confusion_matrix(test_labels, test_preds)
print('Confusion matrix')
print('(rows = true index value for category \n columns = predicted index value for category)\n%s' % cm_data) 
print('\nLabel integers to category words:')
print(labelsdict, '\n')    

# plot confusion matrix to external file
def plot_confusion(cm_data):
    plt.figure()
    selected_cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns_plot = sns.heatmap(cm_data, annot=True, fmt="d", \
	    cmap = selected_cmap, linewidths = 0.5, cbar = False)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation = 0)
    plt.title('Confusion Matrix')
    plt.ylabel('True Index Value for Category')
    plt.xlabel('Predicted Index Value for Category')
    # plt.show() # use if plot to screen is desired
    plt.savefig(FIG_PREFIX + 'fig-confusion-matrix.pdf', orientation ='landscape')
    plt.close()

plot_confusion(cm_data)

if ENCODING == 'WORD_EMBEDDINGS':
    # examine the word embeddings fit to the training data
    # these are located in the initial modeling layer
    embeddings_layer = model.layers[0]
    word_embeddings = embeddings_layer.get_weights()[0]
    print('Word embeddings matrix dimensions (vocab_size, embeddings_dim): ', 
        word_embeddings.shape, '\n')
    
    # create files of vocabulary words and associated embeddings
    # these two files may be used to create visualizations
    # using, for example, the online visualization tool at projector.tensorflow.org
    words_output_file = FIG_PREFIX + 'file_words_out.tsv'
    embeddings_output_file = FIG_PREFIX + 'file_embeddings_out.tsv'
    words_out = io.open(words_output_file, 'w', encoding = 'utf-8')
    embeddings_out = io.open(embeddings_output_file, 'w', encoding = 'utf-8')
    
    for iword in range(1, vocab_size):
        this_word = reverse_word_index[iword]
        this_word_embeddings = word_embeddings[iword,]
        words_out.write(this_word + '\n')
        embeddings_out.write('\t'.join([str(x) for x in this_word_embeddings]) + '\n')
    words_out.close()
    embeddings_out.close() 

print('NEURAL NETWORK MODELING COMPLETE\n')

# repeat information about the TensorFlow environment
print('TensorFlow/Keras environment for this work:')
print('Numpy version: ', np.__version__)
print('TensorFlow version: ', tf.__version__)
print('TensorFlow was built with CUDA: ', tf.test.is_built_with_cuda())
print('TensorFlow running with GPU: ', tf.test.is_gpu_available())
print('Keras version: ', tf.keras.__version__)

if ENCODING == 'WORD_EMBEDDINGS':
    print('\nWord embeddings (TSV file of vectors):  ', embeddings_output_file) 
    print('Vocabulary words (TSV file of metadata): ', words_output_file)
print('Load into http://projector.tensorflow.org/ to view embeddings')

print('\nRUN COMPLETE\n')

log_file.close()