# part1
# Copyright 2024 MIT Introduction to Deep Learning. All Rights Reserved.
#
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of MIT Introduction
# to Deep Learning must reference:
#
# © MIT Introduction to Deep Learning
# http://introtodeeplearning.com
#

#setup 
import comet_ml
COMET_API_KEY = "NAYItnCcXRFTkkSkCfrGwH8jb"
import mitdeeplearning as mdl
import mitdeeplearning.util
import mitdeeplearning.lab1
import numpy as np
import os
import time
import functools
import tensorflow as tf
from tqdm import tqdm
from scipy.io.wavfile import write
import regex as re
import subprocess
import urllib
from IPython import display as ipythondisplay
from IPython.display import Audio

# Download the dataset
songs = mdl.lab1.load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
# print("\nExample song: ")
# print(example_song)

# Convert the ABC notation to audio file and listen to it
mdl.lab1.play_song(example_song)

# # Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# # Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
chars = sorted(list(set(songs_joined)))
print(''.join(chars))

# create a mapping from characters to integers
char2idx = { ch:i for i,ch in enumerate(chars) }
idx2char = { i:ch for i,ch in enumerate(chars) }

# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n}')

def vectorize_string(string):
    chars = sorted(set(string))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    vectorized = np.array([char2idx[ch] for ch in string])
    return vectorized

vectorized_songs = vectorize_string(songs_joined)

print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))

### Batch definition to create training examples ###

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  '''TODO: construct a list of input sequences for the training batch'''
  input_batch = np.array([vectorized_songs[i:i+seq_length] for i in idx])
  '''TODO: construct a list of output sequences for the training batch'''
  output_batch = np.array([vectorized_songs[i+1:i+1+seq_length] for i in idx])

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

# test_args = (vectorized_songs, 10, 2)
# if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
#    not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
#    not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
#    print("======\n[FAIL] could not pass tests")
# else:
#    print("======\n[PASS] passed all tests!")

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


#RNN MODEL

def LSTM(rnn_units):
  return tf.keras.layers.LSTM(
    rnn_units,
    return_sequences=True,
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer='uniform', input_length=None),
    LSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size)
  ])

  return model

model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)

# testing the RNN
model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")


