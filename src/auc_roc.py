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
from tensorflow.keras.regularizers import l1


ENCODING = 'WORD_EMBEDDINGS'
DROPOUT_RATE = 0.5
STDOOUT_TO_FILE = 'True'

FIG_PREFIX = 'AUC_ROC|' + ENCODING + "|DROPOUT=" + str(DROPOUT_RATE) + "|" 

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
nclasses = 1

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
#catslist = list(set(traincats))
catsdict = {'successful': 0, 'failed': 1}
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
print(test_labels[:10])

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

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))

# define model structure with key hyperparameters for multi-class model
def define_model(name, vocab_size, embedding_dim, max_length, nclasses):
    # define the structure of the model
    model = Sequential()
    model = add_layers(name, model, vocab_size, embedding_dim, max_length, nclasses)
    # using sparse_categorical_crossentropy because category labels are integers
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    plot_model(model, to_file = FIG_PREFIX + 'fig-model-structure.png', dpi = 128, show_shapes = True, show_layer_names = True)
    return model    


def add_layers(name, model, vocab_size, embedding_dim, max_length, nclasses):
    if ENCODING == 'ONE-HOT':
        model.add(OneHot(input_dim=vocab_size, input_length=max_length))
    else:
        model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))

    if name == '1D-CNN' :
        model.add(Conv1D(filters = 8, kernel_size = 3, activation = 'relu'))
    elif name == 'DNN' :
        model.add(Flatten())
        model.add(Dense(units = 16, activation = 'relu'))
    elif name == 'LSTM':
        model.add(LSTM(units = 8, activation = 'relu'))
    elif name == 'GRU':
        model.add(GRU(units = 8, activation = 'relu'))

    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))

    if name == '1D-CNN':
        model.add(GlobalMaxPooling1D())

    model.add(Dense(8, activation = 'relu'))

    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(1, activation = 'sigmoid'))
    
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
model1 = define_model('1D-CNN', vocab_size, embedding_dim, max_length, nclasses)
model2 = define_model('DNN', vocab_size, embedding_dim, max_length, nclasses)
model3 = define_model('LSTM', vocab_size, embedding_dim, max_length, nclasses)
model4 = define_model('GRU', vocab_size, embedding_dim, max_length, nclasses)

def train_model(model, name):
    print('\n Training Model: ', name)

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
        
    # examine the predicted values within a precision/recall framework
    test_preds = model1.predict_classes(test_sequence)
    test_preds = list(map(lambda x: x[0], test_preds))
    
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
        metrics.f1_score(test_labels, test_preds, average='weighted')
    print('Test set F1 (weighted average):', \
    	'{:6.4f}'.format(np.round(F1_value, decimals = 4)),'\n')

    print('\n Training Model ends: ', name)
    return model

model1 = train_model(model1, '1D-CNN')
model2 = train_model(model2, 'DNN')
model3 = train_model(model3, 'LSTM')
model4 = train_model(model4, 'GRU')

### Draw AUC_ROC diagram    
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import auc

def plot_roc_curve(fpr,tpr, color, l): 
  plt.plot(fpr,tpr, color, label=l)
  
test_preds_proba1 = model1.predict_proba(test_sequence)
test_preds_proba2 = model2.predict_proba(test_sequence)
test_preds_proba3 = model3.predict_proba(test_sequence)
test_preds_proba4 = model4.predict_proba(test_sequence)

plt.figure(figsize=(10,10))
plt.axis([0,1,0,1]) 
plt.title('AUC_ROC|Dropout_rate={}|Encoding={}'.format(DROPOUT_RATE, ENCODING))
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 



fpr1 , tpr1 , _ = roc_curve ( test_labels , test_preds_proba1)
auc1 = auc(fpr1, tpr1)
plot_roc_curve (fpr1, tpr1, 'b',  '1D-CNN: auc={}'.format(auc1)) 

fpr2 , tpr2 , _ = roc_curve ( test_labels , test_preds_proba2)
auc2 = auc(fpr2, tpr2)
plot_roc_curve (fpr2, tpr2, 'r', 'DNN: auc={}'.format(auc2)) 

fpr3 , tpr3 , _ = roc_curve ( test_labels , test_preds_proba3)
auc3 = auc(fpr3, tpr3)
plot_roc_curve (fpr3, tpr3, 'g', 'LSTM: auc={}'.format(auc3)) 

fpr4 , tpr4 , _ = roc_curve ( test_labels , test_preds_proba4)
auc4 = auc(fpr4, tpr4)
plot_roc_curve (fpr4, tpr4, 'c', 'GRU: auc={}'.format(auc4)) 

plt.legend()
plt.savefig(FIG_PREFIX + 'auc_roc.png', orientation ='landscape')


print('NEURAL NETWORK MODELING COMPLETE\n')

# repeat information about the TensorFlow environment
print('TensorFlow/Keras environment for this work:')
print('Numpy version: ', np.__version__)
print('TensorFlow version: ', tf.__version__)
print('TensorFlow was built with CUDA: ', tf.test.is_built_with_cuda())
print('TensorFlow running with GPU: ', tf.test.is_gpu_available())
print('Keras version: ', tf.keras.__version__)

print('\nRUN COMPLETE\n')

log_file.close()