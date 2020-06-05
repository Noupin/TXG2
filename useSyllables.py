#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import wandb
from wandb.keras import WandbCallback
import re
import tensorflow_datasets as tfds


run = wandb.init(project="sttextgen")
config = run.config
config.batch_size = 512
config.file = r"C:\Datasets\Text\Stranger Things\Season 1\1.txt"
config.maxlen = 128 #Len of sliding window
config.step = 3 #The amount of characters per step
config.epochs = 450
config.charsGen = 500
config.rememberChars = 128

_, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
encoder = info.features['text'].encoder

strDelChars = '[01234567890->:,�♪]'

text = io.open(config.file, encoding='utf-8').read()
text = re.sub(strDelChars, '', text)
text = text.replace("\n\n", "\n").replace("-", "").replace("\n ", "").replace("\n ", "")
encodedText = encoder.encode(text)

with io.open(r"C:\Coding\Python\ML\Text\normalizedText.txt", "w", encoding='utf-8') as newFile:
    newFile.write(text)
chars = sorted(list(set(encodedText)))


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build a sequence for every <config.step>-th character in the text

sentences = []
next_chars = []
for i in range(0, len(encodedText) - config.maxlen, config.step):
    sentences.append(encodedText[i: i + config.maxlen])
    next_chars.append(encodedText[i + config.maxlen])

# build up one-hot encoded input x and output y where x is a character
# in the text y is the next character in the text

x = np.zeros((len(sentences), config.maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#Creating Model
model = keras.models.load_model(r"C:\Coding\Python\ML\Text\Models\STS1_CharGen_512Batch_128Maxlen_128Remem_450Epochs.model")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def printText():
    start_index = random.randint(0, len(encodedText) - config.maxlen - 1)

    for diversity in [0.7]:
        generated = []
        sentence = encodedText[start_index: start_index + config.maxlen]
        generated.extend(sentence)
        print('\n\nDiversity: ' + str(diversity) +"\n\n")
        sys.stdout.write(encoder.decode(generated))
        print("\n\nGenerated:\n")
        for i in range(config.charsGen):
            x_pred = np.zeros((1, config.maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated.append(next_char)
            sentence = sentence[1:]
            sentence.append(next_char)

            sys.stdout.write(encoder.decode([next_char]))
            sys.stdout.flush()
        #print(encoder.decode(generated))

printText()
