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
char2idx = { ch:i for i,ch in enumerate(vocab) }
idx2char = np.array(vocab)

# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n}')

def vectorize_string(string):
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
def build_model(vocab_size, embedding_dim, rnn_units):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer='uniform', input_length=None),
    LSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size)
  ])

  return model

model = build_model(len(vocab), embedding_dim=256, rnn_units=1024)

#checking the model
model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


#Training the model: loss and training operations

#loss function
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss

example_batch_loss = compute_loss(y, pred)

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

  
#Hyperparameter setting and optimization
vocab_size = len(vocab)
params = dict(
  num_training_iterations = 1000,  # Increase this to train longer
  batch_size = 8,  # Experiment between 1 and 64
  seq_length = 100,  # Experiment between 50 and 500
  learning_rate = 5e-3,  # Experiment between 1e-5 and 1e-1
  embedding_dim = 256,
  rnn_units = 1024,  # Experiment between 1 and 2048
)

# Checkpoint location:
checkpoint_dir = '/Users/juliaafonso/Documents/MITlab1/training_checks.weights.h5'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt.weights.h5")

#Comet experiment to track training
def create_experiment():
  if 'experiment' in locals():
    experiment.end()

  experiment = comet_ml.Experiment(
                  api_key="NAYItnCcXRFTkkSkCfrGwH8jb",
                  project_name="MIT_Lab1_Part2")
  
  for param, value in params.items():
    experiment.log_parameter(param, value)
  experiment.flush()

  return experiment

  
#Define optimizer and training operation
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"])
optimizer = tf.keras.optimizers.Adam(params["learning_rate"])


#backprop operations
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
  
  grads = tape.gradient(loss, model.trainable_variables) # gradiant

  optimizer.apply_gradients(zip(grads, model.trainable_variables)) # Applying the gradients to the optimizer
  return loss


#training 

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):

  x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
  loss = train_step(x_batch, y_batch)

  experiment.log_metric("loss", loss.numpy().mean(), step=iter)
  history.append(loss.numpy().mean())
  plotter.plot(history)

  if iter % 100 == 0:
     model.save_weights(checkpoint_prefix)

# Saving the trained model and the weights
model.save_weights(checkpoint_prefix)
experiment.flush()


# Restoring latest checkpoint
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"])
# Restoring weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()


#Prediction of a generated song
def generate_text(model, start_string, generation_length=1000):

  input_eval = [char2idx[s] for s in start_string] # TODO
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      predictions = model(input_eval)
      # Removing the batch dimension
      predictions = tf.squeeze(predictions, 0)

      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

generated_text = generate_text(model, start_string="X", generation_length=1000)

#Play back generated songs

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
  waveform = mdl.lab1.play_song(song)

  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)

    numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
    wav_file_path = f"output_{i}.wav"
    write(wav_file_path, 88200, numeric_data)

    experiment.log_asset(wav_file_path)

experiment.end()