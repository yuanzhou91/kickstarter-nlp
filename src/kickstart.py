#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:24:06 2020

@author: Yuan Zhou
"""


import os

MODELS = ['1D-CNN', 'DNN', 'LSTM', 'GRU']

WORD_ENCODINGS = ['WORD_EMBEDDINGS', 'ONE-HOT']

DROPOUT_RATES = [0, 0.2, 0.4, 0.5]

for model in MODELS:
    for encoding in WORD_ENCODINGS:
        for dropout_rate in DROPOUT_RATES:
            os.system("python ./train_eval_model.py {} {} {} {}".format(model, encoding, dropout_rate, 'True'))

os.system("python auc_roc.py")
os.system("python ./optimize_best_model.py")

